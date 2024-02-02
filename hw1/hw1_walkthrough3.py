from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def hw1_walkthrough3():
    # Load the image "I_Love_New_York.png" into memory
    iheartny_img = Image.open('data/I_Love_New_York.png').convert('RGBA')

    # Create a white rgba background
    white_img = Image.new('RGBA', iheartny_img.size, 'WHITE')
    # Paste the image on the background. Go from the start of the image (0,0)
    white_img.paste(iheartny_img, (0, 0), iheartny_img)
    # Convert the image from RGBA to RGB
    iheartny_img = white_img.convert('RGB')

    # Display the image
    plt.figure()
    plt.imshow(iheartny_img)
    plt.show()

    # Convert the color image into a grayscale image
    gray_iheartny_img = iheartny_img.convert('L')

    # Display the image
    plt.figure()
    plt.imshow(gray_iheartny_img, cmap='gray')
    plt.show()

    # Convert the grayscale image into a binary mask using a threshold value
    threshold = 128  # replace with your threshold value
    binary_mask = np.array(gray_iheartny_img) > threshold

    # Load the image "nyc.png" into memory
    nyc_img = Image.open('data/nyc.png')

    # Resize nyc_img so the image height is 500 pixels
    scale = 500 / nyc_img.height
    small_nyc = nyc_img.resize((round(nyc_img.width * scale), 500))

    # Resize ILoveNY binary_mask so that its height is 400 pixels
    scale = 400 / binary_mask.shape[0]
    resized_mask = np.array(Image.fromarray(binary_mask).resize(
        (round(binary_mask.shape[1] * scale), 400)))

    plt.figure()
    plt.imshow(resized_mask, cmap='gray')
    plt.show()

    # Invert the mask
    iresized_mask = ~resized_mask

    plt.figure()
    plt.imshow(iresized_mask, cmap='gray')
    plt.show()

    # Pad the mask to make it the same size as small_nyc
    pad_height = small_nyc.height - iresized_mask.shape[0]
    pad_width = small_nyc.width - iresized_mask.shape[1]
    iresized_mask = np.pad(iresized_mask, ((pad_height // 2, pad_height -
                           pad_height // 2), (pad_width // 2, pad_width - pad_width // 2)))

    plt.figure()
    plt.imshow(iresized_mask, cmap='gray')
    plt.show()

    # Burn the I <3 NY logo into the Manhattan scene
    red = [255, 0, 0]
    love_small_nyc = np.array(small_nyc)

    red_channel = love_small_nyc[:, :, 0]
    red_channel[iresized_mask] = red[0]
    love_small_nyc[:, :, 0] = red_channel

    # Replace with your code to modify the green and blue channels
    green_channel = love_small_nyc[:, :, 1]
    green_channel[iresized_mask] = red[1]
    love_small_nyc[:, :, 1] = green_channel
    blue_channel = love_small_nyc[:, :, 2]
    blue_channel[iresized_mask] = red[2]
    love_small_nyc[:, :, 2] = blue_channel

    plt.figure()
    plt.imshow(love_small_nyc)
    plt.show()

    # # Save the collage as output_nyc.png
    Image.fromarray(love_small_nyc).save('outputs/nyc_with_logo.png')
