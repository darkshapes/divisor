> A few rough ideas, to be marshaled into something more polished later.

# 1. Interactive Denoising 

Consider denoising as a back-and-forth creative process where rather than having every step be automatic, a user can intervene at any point, allowing them to steer the generation in various ways. For instance, they could temporarily perturb the model's paramters in some way to achieve a desired effect (e.g. with LoRAs), or alter the partially denoised image/video/audio itself. Upon making any such change, the user is presented with a preveiw of the model's new prediction for the final image/video/audio, reflecting the altered trajectory. With modern diffusion models (i.e., rectified flows such as Flux) this can be achieved via a single step x0 prediction (even without distillation), allowing previews to be generated very quickly. This is essentially a free lunch -- ultimate controllability with near "real time" interactivity.   
