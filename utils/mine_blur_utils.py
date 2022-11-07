from utils.utils import *
from models.downsampler import Downsampler


# - blur image - exactly like the NCSR is doing it - 
def get_fft_h(im, blur_type, std_blur=1.6, dim_filter=25):
    assert blur_type in ['uniform_blur', 'gauss_blur'], "blur_type can be or 'uniform' or 'gauss'"
    ch, h, w = im.shape
    fft_h    =  np.zeros((h,w),)
    if blur_type=='uniform_blur':
        t        =  4 # 9//2
        fft_h[h//2-t:h//2+1+t, w//2-t:w//2+1+t]  = 1/81
        fft_h    = np.fft.fft2(np.fft.fftshift(fft_h))
    else: # gauss_blur
        psf = fspecial_gauss(dim_filter, std_blur)
        t = dim_filter // 2 #12 # 25 // 2
        fft_h[h//2-t:h//2+1+t, w//2-t:w//2+1+t]  = psf
        fft_h    =  np.fft.fft2(np.fft.fftshift(fft_h))
    return fft_h


def blur(im, blur_type,std_blur=1.6, dim_filter=25):
    fft_h = get_fft_h(im, blur_type,std_blur,dim_filter)
    imout = np.zeros_like(im)
    for i in range(im.shape[0]):
        im_f    =  np.fft.fft2(im[i, :, :])
        z_f     =  fft_h*im_f # .* of matlab
        z       =  np.real(np.fft.ifft2(z_f))
        imout[i, :, :] = z
    return imout


# - the inverse function H - 
def get_h(n_ch, blur_type, use_fourier, dtype, std_blur, dim_filter):
    assert blur_type in ['uniform_blur', 'gauss_blur'], "blur_type can be or 'uniform' or 'gauss'"
    if not use_fourier:
        return Downsampler(n_ch, 1, blur_type, preserve_size=True).type(dtype)
    return lambda im: torch_blur(im, blur_type, dtype, std_blur, dim_filter)


def torch_blur(im, blur_type, dtype, std_blur, dim_filter):
    fft_h = get_fft_h(torch_to_np(im), blur_type, std_blur, dim_filter)
    fft_h_torch = torch.squeeze(np_to_torch(fft_h),0).cuda()
    im_DFT = torch.fft.fft2(im, dim=(-2,-1)).cuda()
    blurred_im = torch.fft.ifft2(fft_h_torch*im_DFT, dim=(-2,-1)).real
    return blurred_im.type(dtype)
