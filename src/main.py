import numpy as np
import gradio as gr
import cv2

from gscan.core.geometry import homography, projection, slicing
from gscan.core.basic import regulator
from gscan.core.effect import enhancement


def confirm_button_click_handler(img):
    if img is not None:
        pts = slicing.slice_image(img)
        ret = []
        for idx in range(4):
            ret.append(gr.Slider.update(value=pts[idx][0], maximum=img.shape[1], interactive=True))
            ret.append(gr.Slider.update(value=pts[idx][1], maximum=img.shape[0], interactive=True))
        return ret
    else:
        return [0] * 8


def image_sliced_select_handler(evt: gr.SelectData, *pts):
    pick = evt.index
    min_dist = 1e8
    min_idx = 0
    for idx in range(4):
        dist = (pts[idx * 2] - pick[0]) ** 2 + (pts[idx * 2 + 1] - pick[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            min_idx = idx
    ret = []
    for idx in range(4):
        if idx == min_idx:
            ret.append(gr.Slider.update(value=pick[0], interactive=True))
            ret.append(gr.Slider.update(value=pick[1], interactive=True))
        else:
            ret.append(gr.Slider.update())
            ret.append(gr.Slider.update())
    return ret


def p_sliders_update_handler(img, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    if img is not None:
        p0 = np.array([int(p0_x), int(p0_y)])
        p1 = np.array([int(p1_x), int(p1_y)])
        p2 = np.array([int(p2_x), int(p2_y)])
        p3 = np.array([int(p3_x), int(p3_y)])
        img_ret = np.copy(img)
        cv2.line(img_ret, p0, p1, (0, 0, 255), 2)
        cv2.line(img_ret, p1, p2, (0, 0, 255), 2)
        cv2.line(img_ret, p2, p3, (0, 0, 255), 2)
        cv2.line(img_ret, p3, p0, (0, 0, 255), 2)
        return img_ret
    else:
        return img


def correct_button_click_handler(img, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, target_height):
    if img is not None:
        p0 = np.array([int(p0_x), int(p0_y)])
        p1 = np.array([int(p1_x), int(p1_y)])
        p2 = np.array([int(p2_x), int(p2_y)])
        p3 = np.array([int(p3_x), int(p3_y)])
        K = np.array([[1536.1, 0, 959.5], [0, 1535.7, 723.8], [0, 0, 1]])
        p_uv = np.array([p0, p1, p2, p3]).swapaxes(0, 1)

        ratio = projection.calc_real_rect_ratio(K, p_uv)
        target_width = int(target_height * ratio)
        p_target = np.array([[0, target_width, target_width, 0], [0, 0, target_height, target_height]])

        H = homography.get_homography_matrix(p_target, p_uv)
        return homography.homography_correction(img, H, (target_height, target_width), regulator=regulator.GrayCuttingRegulator)
    else:
        return img


with gr.Blocks(title="Document Geometry Correction") as application:
    with gr.Tab("Slicing and Correcting"):
        gr.Markdown("## Upload")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image()
            with gr.Column():
                confirm_button = gr.Button("Confirm")

        gr.Markdown("## Modify")
        with gr.Row():
            with gr.Column():
                image_sliced = gr.Image()
            with gr.Column():
                p_sliders = []
                for idx in range(4):
                    p_sliders.append([
                        gr.Slider(0, 100, 0, step=1, label="p{}.x".format(idx)),
                        gr.Slider(0, 100, 0, step=1, label="p{}.y".format(idx))
                    ])

        gr.Markdown("## Correct")
        with gr.Row():
            with gr.Column():
                image_corrected = gr.Image()
            with gr.Column():
                target_height_slider = gr.Slider(10, 2000, 600, step=1, label="Target Height")
                correct_button = gr.Button("Correct")
                copy_to_enhance_button = gr.Button("Copy to Enhancement Tab")

    with gr.Tab("Enhancing"):
        gr.Markdown("## Enhancement")
        with gr.Row():
            with gr.Column():
                image_pre_enhanced = gr.Image()
            with gr.Column():
                binarization_button = gr.Button("Binarization")

        gr.Markdown("## Result")
        image_enhanced = gr.Image()

    flatten_p_sliders = [item for sublist in p_sliders for item in sublist]
    confirm_button.click(confirm_button_click_handler, inputs=image_input, outputs=flatten_p_sliders)

    image_sliced.select(image_sliced_select_handler, inputs=flatten_p_sliders, outputs=flatten_p_sliders)
    for p_slider in flatten_p_sliders:
        p_slider.change(p_sliders_update_handler, inputs=[image_input] + flatten_p_sliders, outputs=image_sliced, queue=False)

    correct_button.click(correct_button_click_handler, inputs=[image_input] + flatten_p_sliders + [target_height_slider], outputs=image_corrected)
    copy_to_enhance_button.click(lambda img: img, inputs=image_corrected, outputs=image_pre_enhanced)

    binarization_button.click(lambda img: enhancement.binarization(img, 128), inputs=image_pre_enhanced, outputs=image_enhanced)

application.launch()

