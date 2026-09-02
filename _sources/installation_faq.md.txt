# Common Installation Issues

<details>
    <summary><strong>Installation on NVIDIA GeForce RTX 5060 - Blackwell architecture</strong></summary>
    Taken from [Issue #105](https://github.com/RosettaCommons/foundry/issues/105):
    If, when installing any of the models included in foundry, you get an error related to CUDA (see linked issue for examples) try installing a version of torch, torchvision and torchaudio that matches your python and CUDA distribution. For CUDA, choose the next lowest available library if there is not a version that matches your exact CUDA build. 

    For example: 
    ```
    pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    ```
    This is for Python 2.9.1 and CUDA 12.8.

</details>