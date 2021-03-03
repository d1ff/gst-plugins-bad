/*
 * GStreamer
 * Copyright (C) <2017> Philippe Renon <philippe_renon@yahoo.fr>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/**
 * SECTION:element-cameraundistort
 *
 * This element performs camera distortion correction.
 *
 * Camera correction settings are obtained by running through
 * the camera calibration process with the cameracalibrate element.
 *
 * It is possible to do live correction and calibration by chaining
 * a cameraundistort and a cameracalibrate element. The cameracalibrate
 * will send an event with the correction parameters to the the
 * cameraundistort.
 *
 * Based on this tutorial:
 * https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
 *
 * ## Example pipelines
 *
 * |[
 * gst-launch-1.0 -v v4l2src ! videoconvert ! cameraundistort settings="???" !
 * autovideosink
 * ]| will correct camera distortion based on provided settings.
 * |[
 * gst-launch-1.0 -v v4l2src ! videoconvert ! cameraundistort ! cameracalibrate
 * ! autovideosink
 * ]| will correct camera distortion once camera calibration is done.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <vector>

#include "gstcvremap.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <gst/opencv/gstopencvutils.h>

GST_DEBUG_CATEGORY_STATIC(gst_cv_remap_debug);
#define GST_CAT_DEFAULT gst_cv_remap_debug

#define DEFAULT_SHOW_ED TRUE
#define DEFAULT_ALPHA 0.0

enum { PROP_0, PROP_SHOW_ED, PROP_ALPHA, PROP_MAPS };

G_DEFINE_TYPE(GstCvRemap, gst_cv_remap, GST_TYPE_OPENCV_VIDEO_FILTER);

static void gst_cv_remap_dispose(GObject* object);
static void gst_cv_remap_set_property(
    GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec);
static void gst_cv_remap_get_property(
    GObject* object, guint prop_id, GValue* value, GParamSpec* pspec);

static GstCaps* gst_cv_remap_transform_caps(GstBaseTransform* trans,
    GstPadDirection direction, GstCaps* from, GstCaps* filter);
static gboolean gst_cv_remap_set_info(GstOpencvVideoFilter* cvfilter,
    gint in_width, gint in_height, int in_cv_type, gint out_width,
    gint out_height, int out_cv_type);
static GstFlowReturn gst_cv_remap_transform_frame(
    GstOpencvVideoFilter* cvfilter, GstBuffer* frame, cv::Mat img,
    GstBuffer* outframe, cv::Mat outimg);

static void cv_remap_run(GstCvRemap* undist, cv::Mat img, cv::Mat outimg);

/* initialize the cameraundistort's class */
static void gst_cv_remap_class_init(GstCvRemapClass* klass)
{
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstOpencvVideoFilterClass* opencvfilter_class
        = GST_OPENCV_VIDEO_FILTER_CLASS(klass);
    GstBaseTransformClass* trans_class = (GstBaseTransformClass*)klass;

    GstCaps* caps;
    GstPadTemplate* templ;

    gobject_class->dispose = gst_cv_remap_dispose;
    gobject_class->set_property = gst_cv_remap_set_property;
    gobject_class->get_property = gst_cv_remap_get_property;

    opencvfilter_class->cv_set_caps = gst_cv_remap_set_info;
    opencvfilter_class->cv_trans_func = gst_cv_remap_transform_frame;

    g_object_class_install_property(gobject_class, PROP_SHOW_ED,
        g_param_spec_boolean("undistort", "Apply camera corrections",
            "Apply camera corrections", DEFAULT_SHOW_ED,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_ALPHA,
        g_param_spec_float("alpha", "Pixels",
            "Show all pixels (1), only valid ones (0) or something in between",
            0.0, 1.0, DEFAULT_ALPHA,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_MAPS,
        g_param_spec_string("maps", "Maps",
            "Maps path (stored in cv filestorage format)", NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata(element_class, "cvremap",
        "Filter/Effect/Video", "Performs cv remap",
        "Vladislav Bortnikov <bortnikov.vladislav@e-sakha.ru>");

    /* add sink and source pad templates */
    caps = gst_opencv_caps_from_cv_image_type(CV_16UC1);
    gst_caps_append(caps, gst_opencv_caps_from_cv_image_type(CV_8UC4));
    gst_caps_append(caps, gst_opencv_caps_from_cv_image_type(CV_8UC3));
    gst_caps_append(caps, gst_opencv_caps_from_cv_image_type(CV_8UC1));
    templ = gst_pad_template_new(
        "sink", GST_PAD_SINK, GST_PAD_ALWAYS, gst_caps_ref(caps));
    gst_element_class_add_pad_template(element_class, templ);
    templ = gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps);
    gst_element_class_add_pad_template(element_class, templ);

    trans_class->transform_caps
        = GST_DEBUG_FUNCPTR(gst_cv_remap_transform_caps);
}

/* initialize the new element
 * initialize instance structure
 */
static void gst_cv_remap_init(GstCvRemap* undist)
{
    undist->showUndistorted = DEFAULT_SHOW_ED;
    undist->alpha = DEFAULT_ALPHA;

    undist->doRemap = FALSE;
    undist->settingsChanged = FALSE;

    undist->map1 = cv::UMat();
    undist->map2 = cv::UMat();

    undist->mapsPath = NULL;
    undist->pad_sink_height = 0;
    undist->pad_sink_width = 0;

    gst_opencv_video_filter_set_in_place(
        GST_OPENCV_VIDEO_FILTER_CAST(undist), FALSE);
}

static void gst_cv_remap_dispose(GObject* object)
{
    GstCvRemap* undist = GST_CV_REMAP(object);

    g_free(undist->mapsPath);

    G_OBJECT_CLASS(gst_cv_remap_parent_class)->dispose(object);
}

static void gst_cv_remap_set_property(
    GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec)
{
    GstCvRemap* undist = GST_CV_REMAP(object);
    const char* str;

    switch (prop_id) {
    case PROP_SHOW_ED:
        undist->showUndistorted = g_value_get_boolean(value);
        break;
    case PROP_ALPHA:
        undist->alpha = g_value_get_float(value);
        break;
    case PROP_MAPS:
        if (undist->mapsPath) {
            g_free(undist->mapsPath);
            undist->mapsPath = NULL;
        }
        str = g_value_get_string(value);
        if (str)
            undist->mapsPath = g_strdup(str);
        if (undist->mapsPath) {
            cv::FileStorage fs(undist->mapsPath, 0);
            fs["map_a"].mat().copyTo(undist->map1);
            GST_WARNING("READ MAP A");
            fs["map_b"].mat().copyTo(undist->map2);
            GST_WARNING("READ MAP B");
            fs.release();
        }
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }

    undist->settingsChanged = TRUE;
    gst_base_transform_reconfigure_src(GST_BASE_TRANSFORM_CAST(undist));
}

static void gst_cv_remap_get_property(
    GObject* object, guint prop_id, GValue* value, GParamSpec* pspec)
{
    GstCvRemap* undist = GST_CV_REMAP(object);

    switch (prop_id) {
    case PROP_SHOW_ED:
        g_value_set_boolean(value, undist->showUndistorted);
        break;
    case PROP_ALPHA:
        g_value_set_float(value, undist->alpha);
        break;
    case PROP_MAPS:
        g_value_set_string(value, undist->mapsPath);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

gboolean gst_cv_remap_set_info(GstOpencvVideoFilter* cvfilter, gint in_width,
    gint in_height, G_GNUC_UNUSED int in_cv_type, gint out_width,
    gint out_height, G_GNUC_UNUSED int out_cv_type)
{
    GstCvRemap* undist = GST_CV_REMAP(cvfilter);

    undist->imageSize = cv::Size(in_width, in_height);
    GST_WARNING("out_size=%d,%d", out_width, out_height);

    return TRUE;
}

/*
 * Performs the camera undistort
 */
static GstFlowReturn gst_cv_remap_transform_frame(
    GstOpencvVideoFilter* cvfilter, G_GNUC_UNUSED GstBuffer* frame,
    cv::Mat img, G_GNUC_UNUSED GstBuffer* outframe, cv::Mat outimg)
{
    GstCvRemap* undist = GST_CV_REMAP(cvfilter);

    cv_remap_run(undist, img, outimg);

    return GST_FLOW_OK;
}

static void cv_remap_run(GstCvRemap* undist, cv::Mat img, cv::Mat outimg)
{
    /* TODO is settingsChanged handling thread safe ? */
    if (undist->settingsChanged) {
        undist->settingsChanged = FALSE;
        undist->doRemap = FALSE;
        /* TODO: validate mat's propely */
        undist->doRemap = !undist->map1.empty() && !undist->map2.empty();
    }
    // cv::UMat uimg = img.getUMat(cv::ACCESS_READ),
    // uout = outimg.getUMat(cv::ACCESS_WRITE);

    if (undist->showUndistorted && undist->doRemap) {
        /* do the undistort */

        // GST_WARNING("input img %d,%d; output img %d,%d", img.cols, img.rows,
        // outimg.cols, outimg.rows);
        cv::remap(img.getUMat(cv::ACCESS_READ
                      | cv::ACCESS_FAST), // cv::USAGE_ALLOCATE_DEVICE_MEMORY),
            outimg.getUMat(cv::ACCESS_WRITE),
            // cv::USAGE_ALLOCATE_DEVICE_MEMORY),
            undist->map1, undist->map2, cv::INTER_NEAREST,
            cv::BORDER_TRANSPARENT);

        // uout.copyTo(outimg);
    } else {
        /* FIXME should use pass through to avoid this copy when not
         * undistorting */
        img.copyTo(outimg);
    }
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
gboolean gst_cv_remap_plugin_init(GstPlugin* plugin)
{
    /* debug category for filtering log messages */
    GST_DEBUG_CATEGORY_INIT(
        gst_cv_remap_debug, "cvremap", 0, "Performs remap on images");

    return gst_element_register(
        plugin, "cvremap", GST_RANK_NONE, GST_TYPE_CV_REMAP);
}
static void gst_cv_remap_calculate_dimensions(GstCvRemap* filter,
    GstPadDirection direction, gint in_width, gint in_height, gint* out_width,
    gint* out_height)
{
    GST_LOG_OBJECT(filter,
        "Calculate dimensions, in_width: %" G_GINT32_FORMAT
        " in_height: %" G_GINT32_FORMAT " pad sink width: %" G_GINT32_FORMAT
        " pad sink height: %" G_GINT32_FORMAT " cols: %" G_GINT32_FORMAT
        ", rows: %" G_GINT32_FORMAT ", direction: %d",
        in_width, in_height, filter->pad_sink_width, filter->pad_sink_height,
        filter->map1.cols, filter->map1.rows, direction);

    if (direction == GST_PAD_SINK) {
        *out_width = filter->map1.cols;
        *out_height = filter->map1.rows;

        filter->pad_sink_width = in_width;
        filter->pad_sink_height = in_height;
    } else {
        if (filter->pad_sink_width > 0) {
            *out_width = filter->pad_sink_width;
        } else {
            *out_width = in_width;
        }
        if (filter->pad_sink_height > 0) {
            *out_height = filter->pad_sink_height;
        } else {
            *out_height = in_height;
        }
    }

    GST_LOG_OBJECT(filter,
        "Calculated dimensions: width %" G_GINT32_FORMAT
        " => %" G_GINT32_FORMAT ", height %" G_GINT32_FORMAT
        " => %" G_GINT32_FORMAT " direction: %d",
        in_width, *out_width, in_height, *out_height, direction);
}

static GstCaps* gst_cv_remap_transform_caps(GstBaseTransform* trans,
    GstPadDirection direction, GstCaps* caps, GstCaps* filter_caps)
{
    GstCvRemap* cvremap = GST_CV_REMAP(trans);

    GstCaps* ret;
    gint width, height;
    guint i;

    ret = gst_caps_copy(caps);

    GST_OBJECT_LOCK(cvremap);

    for (i = 0; i < gst_caps_get_size(ret); i++) {
        GstStructure* structure = gst_caps_get_structure(ret, i);

        if (gst_structure_get_int(structure, "width", &width)
            && gst_structure_get_int(structure, "height", &height)) {
            gint out_width, out_height;
            gst_cv_remap_calculate_dimensions(
                cvremap, direction, width, height, &out_width, &out_height);
            gst_structure_set(structure, "width", G_TYPE_INT, out_width,
                "height", G_TYPE_INT, out_height, NULL);
        }
    }

    GST_OBJECT_UNLOCK(cvremap);

    if (filter_caps) {
        GstCaps* intersection;

        GST_DEBUG_OBJECT(
            cvremap, "Using filter caps %" GST_PTR_FORMAT, filter_caps);

        intersection = gst_caps_intersect_full(
            filter_caps, ret, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(ret);
        ret = intersection;

        GST_DEBUG_OBJECT(cvremap, "Intersection %" GST_PTR_FORMAT, ret);
    }

    return ret;
}
