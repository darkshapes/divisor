# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

import gradio as gr
from nnll.init_gpu import device

from divisor.mmada import app
from divisor.mmada.loading import load_model
from divisor.mmada.spec import configs

css_styles = """
.gradio-container{font-family:'IBM Plex Sans',sans-serif;margin:auto;}
.gr-input {background:#f9f9f9 !important;border:1px solid #e0e0e0 !important;}
.gr-output{background:#f0f0f0 !important;border:1px solid #d0d0d0 !important;}

.highlighted-text span{
    padding:2px 4px;border-radius:4px;margin:1px 2px;display:inline-block;line-height:1.6;
}

footer{display:none !important}

#live-update-scrollable-box {
    max-height: 800px; /* 您可以根据需要调整这个最大高度，例如 '300px', '50vh' 等 */
    overflow-y: auto !important; /* 当内容超出 max-height 时显示垂直滚动条 */
    display: block; /* 确保元素是块级元素，以便 max-height 生效 */

}
#think_btn {
    background-color: #f3f4f6 !important;
    border: 1px solid #d0d0d0 !important;
    color: #111827 !important;
    font-size: 16px !important;
    font-weight: bold !important;
}
#think_btn:hover {
    background-color: #e0e0e0 !important;
    border: 1px solid #c0c0c0 !important;
    color: #222 !important;
}
#think_btn:active {
    background-color: #2563eb !important;
    border: 1px solid #b0b0b0 !important;
    color: white !important;
}
"""


def toggle_thinking_mode_lm(current_thinking_mode):
    new_state = not current_thinking_mode
    new_label = "Thinking Mode ✅" if new_state else "Thinking Mode ❌"
    return new_state, gr.update(value=new_label)


def toggle_thinking_mode_mmu(current_thinking_mode):
    new_state = not current_thinking_mode
    new_label = "Thinking Mode ✓" if new_state else "Thinking Mode ✗"
    return new_state, gr.update(value=new_label)


color_map_config = {
    "MASK": "lightgrey",
    "GEN": "#DCABFA",
}

model_choices = ["model.mldm.mmada"] + ["model.mldm.mmada:" + n for n in configs["model.mldm.mmada"] if n != "*"]
with gr.Blocks(css=css_styles) as demo:
    thinking_mode_lm = gr.State(False)
    thinking_mode_mmu = gr.State(False)

    gr.Markdown("### Select Model")
    with gr.Row():
        model_select_radio = gr.Radio(label="", choices=model_choices, value=model_choices[0])
        model_load_status_box = gr.Textbox(label="Model Load Status", interactive=False, lines=3, max_lines=5)

    gr.Markdown("## Text Generation")
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input_box_lm = gr.Textbox(
                label="Enter your prompt:",
                lines=3,
                value="A rectangular prism has a length of 5 units, a width of 4 units, and a height of 3 units. What is the volume of the prism?",
            )
            think_button_lm = gr.Button("Toggle Thinking", elem_id="think_btn")
            with gr.Accordion("Generation Parameters", open=True):
                with gr.Row():
                    gen_length_slider_lm = gr.Slider(minimum=8, maximum=1024, value=512, step=64, label="Generation Length", info="Number of tokens to generate.")
                    steps_slider_lm = gr.Slider(minimum=1, maximum=512, value=256, step=32, label="Total Sampling Steps", info="Must be divisible by (gen_length / block_length).")
                with gr.Row():
                    block_length_slider_lm = gr.Slider(minimum=8, maximum=1024, value=128, step=32, label="Block Length", info="gen_length must be divisible by this.")
                    remasking_dropdown_lm = gr.Dropdown(choices=["low_confidence", "random"], value="low_confidence", label="Remasking Strategy")
                with gr.Row():
                    cfg_scale_slider_lm = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="CFG Scale", info="Classifier-Free Guidance. 0 disables it.")
                    temperature_slider_lm = gr.Slider(
                        minimum=0.0, maximum=2.0, value=1, step=0.05, label="Temperature", info="Controls randomness via Gumbel noise. 0 is deterministic."
                    )

            with gr.Row():
                run_button_ui_lm = gr.Button("Generate Sequence", variant="primary", scale=3)
                clear_button_ui_lm = gr.Button("Clear Outputs", scale=1)

        with gr.Column(scale=3):
            output_visualization_box_lm = gr.HighlightedText(
                label="Live Generation Process",
                show_legend=True,
                color_map=color_map_config,
                combine_adjacent=False,
                interactive=False,
                elem_id="live-update-scrollable-box",
            )
            output_final_text_box_lm = gr.Textbox(label="Final Output", lines=8, interactive=False, show_copy_button=True)

    gr.Examples(
        examples=[
            [
                model_choices[0],
                "A rectangular prism has a length of 5 units, a width of 4 units, and a height of 3 units. What is the volume of the prism?",
                256,
                512,
                128,
                1,
                0,
                "low_confidence",
                False,
            ],
            [
                model_choices[0],
                "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?",
                256,
                512,
                64,
                1,
                0,
                "low_confidence",
                False,
            ],
        ],
        inputs=[
            model_select_radio,
            prompt_input_box_lm,
            steps_slider_lm,
            gen_length_slider_lm,
            block_length_slider_lm,
            temperature_slider_lm,
            cfg_scale_slider_lm,
            remasking_dropdown_lm,
            thinking_mode_lm,
        ],
        outputs=[output_visualization_box_lm, output_final_text_box_lm],
        fn=app.generate_viz_wrapper_lm,
    )

    # gr.Markdown("---")
    # gr.Markdown("## Part 2. Multimodal Understanding")
    # with gr.Row():
    #     with gr.Column(scale=2):
    #         prompt_input_box_mmu = gr.Textbox(label="Enter your prompt:", lines=3, value="Please describe this image in detail.")
    #         think_button_mmu = gr.Button("Toggle Thinking", elem_id="think_btn")
    #         with gr.Accordion("Generation Parameters", open=True):
    #             with gr.Row():
    #                 gen_length_slider_mmu = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Generation Length", info="Number of tokens to generate.")
    #                 steps_slider_mmu = gr.Slider(minimum=1, maximum=512, value=256, step=32, label="Total Sampling Steps", info="Must be divisible by (gen_length / block_length).")
    #             with gr.Row():
    #                 block_length_slider_mmu = gr.Slider(minimum=32, maximum=1024, value=128, step=32, label="Block Length", info="gen_length must be divisible by this.")
    #                 remasking_dropdown_mmu = gr.Dropdown(choices=["low_confidence", "random"], value="low_confidence", label="Remasking Strategy")
    #             with gr.Row():
    #                 cfg_scale_slider_mmu = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="CFG Scale", info="Classifier-Free Guidance. 0 disables it.")
    #                 temperature_slider_mmu = gr.Slider(
    #                     minimum=0.0, maximum=2.0, value=1, step=0.05, label="Temperature", info="Controls randomness via Gumbel noise. 0 is deterministic."
    #                 )

    #         with gr.Row():
    #             image_upload_box = gr.Image(type="pil", label="Upload Image")

    #         with gr.Row():
    #             run_button_ui_mmu = gr.Button("Generate Description", variant="primary", scale=3)
    #             clear_button_ui_mmu = gr.Button("Clear Outputs", scale=1)

    #     with gr.Column(scale=3):
    #         gr.Markdown("## Live Generation Process")
    #         output_visualization_box_mmu = gr.HighlightedText(
    #             label="Token Sequence (Live Update)",
    #             show_legend=True,
    #             color_map=color_map_config,
    #             combine_adjacent=False,
    #             interactive=False,
    #             elem_id="live-update-scrollable-box",
    #         )
    #         gr.Markdown("## Final Generated Text")
    #         output_final_text_box_mmu = gr.Textbox(label="Final Output", lines=8, interactive=False, show_copy_button=True)

    # gr.Examples(
    #     examples=[
    #         ["mmu_validation_2/sunflower.jpg", "Please describe this image in detail.", 256, 512, 128, 1, 0, "low_confidence"],
    #         ["mmu_validation_2/woman.jpg", "Please describe this image in detail.", 256, 512, 128, 1, 0, "low_confidence"],
    #     ],
    #     inputs=[
    #         image_upload_box,
    #         prompt_input_box_mmu,
    #         steps_slider_mmu,
    #         gen_length_slider_mmu,
    #         block_length_slider_mmu,
    #         temperature_slider_mmu,
    #         cfg_scale_slider_mmu,
    #         remasking_dropdown_mmu,
    #     ],
    #     outputs=[output_visualization_box_mmu, output_final_text_box_mmu],
    #     fn=app.generate_viz_wrapper,
    # )

    think_button_lm.click(fn=toggle_thinking_mode_lm, inputs=[thinking_mode_lm], outputs=[thinking_mode_lm, think_button_lm])
    # think_button_mmu.click(fn=toggle_thinking_mode_mmu, inputs=[thinking_mode_mmu], outputs=[thinking_mode_mmu, think_button_mmu])

    def initialize_model():
        default_model_id = model_choices[0]
        try:
            load_model(default_model_id, device=device)
            status = f"Model '{default_model_id}' loaded successfully."
        except Exception as e:
            status = f"Error loading model '{default_model_id}': {str(e)}"
        return default_model_id, status

    demo.load(fn=initialize_model, inputs=None, outputs=[model_select_radio, model_load_status_box], queue=True)

    def clear_outputs():
        return None, None  # Clear visualization and final text

    clear_button_ui_lm.click(fn=clear_outputs, inputs=None, outputs=[output_visualization_box_lm, output_final_text_box_lm], queue=False)
    # clear_button_ui_mmu.click(fn=clear_outputs, inputs=None, outputs=[image_upload_box, output_visualization_box_mmu, output_final_text_box_mmu], queue=False)

    run_button_ui_lm.click(
        fn=app.generate_viz_wrapper_lm,
        inputs=[
            model_select_radio,
            prompt_input_box_lm,
            steps_slider_lm,
            gen_length_slider_lm,
            block_length_slider_lm,
            temperature_slider_lm,
            cfg_scale_slider_lm,
            remasking_dropdown_lm,
            thinking_mode_lm,
        ],
        outputs=[output_visualization_box_lm, output_final_text_box_lm],
    )

    # run_button_ui_mmu.click(
    #     fn=app.generate_viz_wrapper,
    #     inputs=[
    #         image_upload_box,
    #         prompt_input_box_mmu,
    #         steps_slider_mmu,
    #         gen_length_slider_mmu,
    #         block_length_slider_mmu,
    #         temperature_slider_mmu,
    #         cfg_scale_slider_mmu,
    #         remasking_dropdown_mmu,
    #         thinking_mode_mmu,
    #     ],
    #     outputs=[output_visualization_box_mmu, output_final_text_box_mmu],
    # )


def main():
    print(f"Starting Gradio App. Attempting to use device: {device.type}")
    demo.launch(share=True)


if __name__ == "__main__":
    main()
