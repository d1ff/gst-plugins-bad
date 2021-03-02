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

    undist->map1 = cv::Mat();
    undist->map2 = cv::Mat();

    undist->mapsPath = NULL;

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
            undist->map1 = fs["map_a"].mat();
            GST_WARNING("READ MAP A");
            undist->map2 = fs["map_b"].mat();
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

    if (undist->showUndistorted && undist->doRemap) {
        /* do the undistort */

        // GST_WARNING("input img %d,%d; output img %d,%d", img.cols, img.rows,
        // outimg.cols, outimg.rows);
        cv::Mat outRoi(outimg, cv::Range(0, undist->map1.rows),
            cv::Range(0, undist->map1.cols));
        cv::remap(img, outRoi, undist->map1, undist->map2, cv::INTER_LINEAR,
            cv::BORDER_TRANSPARENT);

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

static inline gint gst_cv_remap_transform_dimension(gint val, gint delta)
{
    gint64 new_val = (gint64)val - (gint64)delta;

    new_val = CLAMP(new_val, 1, G_MAXINT);

    return (gint)new_val;
}

static gboolean gst_cv_remap_transform_dimension_value(
    const GValue* src_val, gint delta, GValue* dest_val)
{
    gboolean ret = TRUE;

    g_value_init(dest_val, G_VALUE_TYPE(src_val));

    if (G_VALUE_HOLDS_INT(src_val)) {
        gint ival = g_value_get_int(src_val);

        ival = gst_cv_remap_transform_dimension(ival, delta);
        g_value_set_int(dest_val, ival);
    } else if (GST_VALUE_HOLDS_INT_RANGE(src_val)) {
        gint min = gst_value_get_int_range_min(src_val);
        gint max = gst_value_get_int_range_max(src_val);

        // min = #;gst_cv_remap_transform_dimension(min, delta);
        max = gst_cv_remap_transform_dimension(max, delta);
        if (min >= max) {
            ret = FALSE;
            g_value_unset(dest_val);
        } else {
            gst_value_set_int_range(dest_val, min, max);
        }
    } else if (GST_VALUE_HOLDS_LIST(src_val)) {
        guint i;

        for (i = 0; i < gst_value_list_get_size(src_val); ++i) {
            const GValue* list_val;
            GValue newval = {
                0,
            };

            list_val = gst_value_list_get_value(src_val, i);
            if (gst_cv_remap_transform_dimension_value(
                    list_val, delta, &newval))
                gst_value_list_append_value(dest_val, &newval);
            g_value_unset(&newval);
        }

        if (gst_value_list_get_size(dest_val) == 0) {
            g_value_unset(dest_val);
            ret = FALSE;
        }
    } else {
        g_value_unset(dest_val);
        ret = FALSE;
    }

    return ret;
}

static GstCaps* gst_cv_remap_transform_caps(GstBaseTransform* trans,
    GstPadDirection direction, GstCaps* from, GstCaps* filter)
{
    GstCvRemap* cvremap = GST_CV_REMAP(trans);

    GstCaps *to, *ret;
    GstCaps* templ;
    GstStructure* structure;
    GstPad* other;
    to = gst_caps_new_empty();
    guint i;

    for (i = 0; i < gst_caps_get_size(from); ++i) {
        structure = gst_structure_copy(gst_caps_get_structure(from, i));
        if (!cvremap->map1.empty()) {
            GValue w_val = {
                0,
            };
            GValue h_val = {
                0,
            };
            const GValue* v;

            gint dw = 0;
            gint dh = 0;

            if (direction == GST_PAD_SINK) {
                dw -= cvremap->map1.cols - 3072;
                dh -= cvremap->map1.rows - 2048;
            } else {
                dw += cvremap->map1.cols + 3072;
                dh += cvremap->map1.rows + 2048;
            }

            v = gst_structure_get_value(structure, "width");
            if (!gst_cv_remap_transform_dimension_value(v, dw, &w_val)) {
                GST_WARNING_OBJECT(cvremap,
                    "could not transform width value with dw=%d"
                    ", caps structure=%" GST_PTR_FORMAT,
                    dw, structure);
                goto bail;
            }
            gst_structure_set_value(structure, "width", &w_val);

            v = gst_structure_get_value(structure, "height");
            if (!gst_cv_remap_transform_dimension_value(v, dh, &h_val)) {
                g_value_unset(&w_val);
                GST_WARNING_OBJECT(cvremap,
                    "could not transform height value with dh=%d"
                    ", caps structure=%" GST_PTR_FORMAT,
                    dh, structure);
                goto bail;
            }
            gst_structure_set_value(structure, "height", &h_val);
            g_value_unset(&w_val);
            g_value_unset(&h_val);
        }
        gst_caps_append_structure(to, structure);
    }

    /* filter against set allowed caps on the pad */
    other = (direction == GST_PAD_SINK) ? trans->srcpad : trans->sinkpad;
    templ = gst_pad_get_pad_template_caps(other);
    ret = gst_caps_intersect(to, templ);
    gst_caps_unref(to);
    gst_caps_unref(templ);

    GST_WARNING_OBJECT(cvremap,
        "direction %d, transformed %" GST_PTR_FORMAT " to %" GST_PTR_FORMAT,
        direction, from, ret);

    if (ret && filter) {
        GstCaps* intersection;

        GST_WARNING_OBJECT(
            cvremap, "Using filter caps %" GST_PTR_FORMAT, filter);
        intersection
            = gst_caps_intersect_full(filter, ret, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(ret);
        ret = intersection;
        GST_WARNING_OBJECT(cvremap, "Intersection %" GST_PTR_FORMAT, ret);
    }

    return ret;
bail : {
    gst_structure_free(structure);
    gst_caps_unref(to);
    to = gst_caps_new_empty();
    return to;
}
}
