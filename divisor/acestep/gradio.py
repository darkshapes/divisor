"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os


def main(
    checkpoint_path="",
    server_name="127.0.0.1",
    port=7865,
    share=False,
    bf16=True,
    torch_compile=False,
    cpu_offload=False,
    overlapped_decode=False,
):
    """Main function to launch the ACE Step pipeline demo.\n
    :param checkpoint_path: Path to the checkpoint directory. Downloads automatically if empty.
    :param server_name: The server name to use for the Gradio app.
    :param port: The port to use for the Gradio app.
    :param share: Whether to create a public, shareable link for the Gradio app.
    :param bf16: Whether to use bfloat16 precision. Turn off if using MPS
    :param torch_compile: Whether to use torch.compile.
    :param cpu_offload: Whether to use CPU offloading (only load current stage's model to GPU.
    :param overlapped_decode: Whether to use overlapped decoding (run dcae and vocoder using sliding windows.
    """

    from divisor.acestep.pipeline_ace_step import ACEStepPipeline
    from divisor.acestep.ui.components import create_main_demo_ui
    from divisor.acestep.data_sampler import DataSampler

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
        load_data_func=data_sampler.load_json,
    )

    demo.launch(server_name=server_name, server_port=port, share=share)


if __name__ == "__main__":
    main()
