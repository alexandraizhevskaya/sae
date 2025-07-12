from pydantic import BaseModel


class TrainParams(BaseModel):
    device: str
    batch_img: int
    batch_sae: int
    input_d: int
    latent_n: int
    epochs_sae: int
    lambda_: float
    alpha: float
    top_k: int
    num_b: int
    lr_sae: float
    weight_decay: float
    img_path: str
    seed: int
    save_file: str