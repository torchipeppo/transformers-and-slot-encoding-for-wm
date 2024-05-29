import other_archs.steve_singh.train as trainer

import torch
import hydra

def hydraconf_autohandle_torch_device(f):
    def wrapper(conf):
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        f(conf)
    return wrapper


@hydra.main(config_path="config", config_name="STEVE")
@hydraconf_autohandle_torch_device
def main(conf):
    trainer.main(conf)

main()
