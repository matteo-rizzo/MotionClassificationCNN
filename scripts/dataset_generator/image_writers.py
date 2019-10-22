import os
from PIL import Image, ImageDraw, ImageFilter
# from tqdm import tqdm


def blur_image(image, radius=2, fallback_mode='L'):
    """
    Blur an image using Gaussian Blur with specified radius. Other possible blur filters are described
    here: https://pillow.readthedocs.io/en/3.1.x/reference/ImageFilter.html
    IMPORTANT: filters are not applicable to mode '1'. If image mode is '1' image will be converted to
    specified fallback_mode, which is L by default
    More details on modes
    here: https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-modes

    :param image: PIL image object
    :param radius: radius of blur
    :param fallback_mode: mode to convert image to if current mode is 1
    :return: blurred image with size unchanged
    """

    if image.mode == '1':
        image = image.convert(fallback_mode)
    return image.filter(ImageFilter.GaussianBlur(radius))


def write_coherent_3D(data, path, blur, img_res, img_res_r, r, center, img_name):
    # for idx_t, coherent_trial in tqdm(enumerate(data), ascii=True, desc='Writing coherent images...', ncols=100):
    for idx_t, coherent_trial in enumerate(data):
        image_arr = []

        for idx_i, interval in enumerate(range(len(coherent_trial[0, 0]))):
            img = Image.new(mode='L', size=img_res, color='white')
            draw = ImageDraw.Draw(img)

            for coordinate in range(len(coherent_trial[0])):
                x = coherent_trial[0, coordinate, interval] + center
                y = coherent_trial[1, coordinate, interval] + center
                draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

            del draw

            img = img.resize((img_res_r, img_res_r))

            if blur:
                img = blur_image(img)

            image_arr.append(img)

        Image.merge("RGBA", [image_arr[0].getchannel(0), image_arr[1].getchannel(0), image_arr[2].getchannel(0),
                             image_arr[3].getchannel(0)]) \
            .save(fp=os.path.join(path, 'coherent', 't{idx_t}{c}'.format(
            idx_t=idx_t,
            c=img_name)),
                  quality=100,
                  format='png')


def write_noise_3D(data, path, blur, img_res, img_res_r, r, center, img_name):
    # for idx_t, noise_trial in tqdm(enumerate(data), ascii=True, desc='Writing noise images...', ncols=100):
    for idx_t, noise_trial in enumerate(data):
        image_arr = []

        for idx_i, interval in enumerate(range(len(noise_trial[0, 0]))):
            img = Image.new(mode='L', size=img_res, color='white')
            draw = ImageDraw.Draw(img)

            for coordinate in range(len(noise_trial[0])):
                x = noise_trial[0, coordinate, interval] + center
                y = noise_trial[1, coordinate, interval] + center
                draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

            del draw

            img = img.resize((img_res_r, img_res_r))

            if blur:
                img = blur_image(img)

            image_arr.append(img)

        Image.merge(mode='RGBA',
                    bands=[image_arr[0].getchannel(0), image_arr[1].getchannel(0), image_arr[2].getchannel(0),
                           image_arr[3].getchannel(0)]) \
            .save(fp=os.path.join(path, 'noise', 't{idx_t}{n}'.format(
            idx_t=idx_t,
            n=img_name)),
                  quality=100,
                  format='png')


def write_coherent_2D(data, path, blur, img_res, img_res_r, r, center, img_name):
    # for idx_t, coherent_trial in tqdm(enumerate(data), ascii=True, desc='Writing coherent images...', ncols=100):
    for idx_t, coherent_trial in enumerate(data):

        for idx_i, interval in enumerate(range(len(coherent_trial[0, 0]))):
            img = Image.new(mode='1', size=img_res, color='white')
            draw = ImageDraw.Draw(img)

            for coordinate in range(len(coherent_trial[0])):
                x = coherent_trial[0, coordinate, interval] + center
                y = coherent_trial[1, coordinate, interval] + center
                draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

            del draw

            img = img.resize((img_res_r, img_res_r))

            if blur:
                img = blur_image(img)

            img.save(os.path.join(path, 'coherent', 't{idx_t}-i{idx_i}{c}'.format(
                idx_t=idx_t,
                idx_i=idx_i,
                c=img_name)),
                     quality=100)


def write_noise_2D(data, path, blur, img_res, img_res_r, r, center, img_name):
    # for idx_t, noise_trial in tqdm(enumerate(data), ascii=True, desc='Writing noise images...', ncols=100):
    for idx_t, noise_trial in enumerate(data):

        for idx_i, interval in enumerate(range(len(noise_trial[0, 0]))):
            img = Image.new(mode='1', size=img_res, color='white')
            draw = ImageDraw.Draw(img)

            for coordinate in range(len(noise_trial[0])):
                x = noise_trial[0, coordinate, interval] + center
                y = noise_trial[1, coordinate, interval] + center
                draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

            del draw

            img = img.resize((img_res_r, img_res_r))

            if blur:
                img = blur_image(img)

            img.save(os.path.join(path, 'noise', 't{idx_t}-i{idx_i}{c}'.format(
                idx_t=idx_t,
                idx_i=idx_i,
                c=img_name)),
                     quality=100)
