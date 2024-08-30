# this is just a quick code to check pixelwise differences between generated images
# I use this to check the outputs from stable diffusion with different VAEs


from PIL import Image, ImageChops,  ImageOps, ImageStat

import os

path1 = r"C:\Users\kukai\Downloads\New folder\grid-0004 (1).png"
path2 = r"C:\Users\kukai\Downloads\New folder\grid-0003 (3).png"


img = Image.open(path1).convert("RGBA")
img2 = Image.open(path2).convert("RGBA")

# Calculate the difference between the images
diff = ImageChops.difference(img, img2)

alpha = diff.convert("L")

# Create a new image with a transparent background
transparent_diff = Image.new("RGBA", img.size, (255, 255, 255, 0))

# Paste the difference onto the transparent background using the alpha mask
transparent_diff.paste(diff, (0, 0), alpha)



# Show the result

if diff.getbbox():
    print("Differences exist between the images.")
    transparent_diff.show()

else:
    print("No differences found between the images.")

from collections import Counter

# Calculate the number of differing pixels
num_diff_pixels = sum(alpha.getdata())

# Calculate the average difference per channel (R, G, B, A)
stat = ImageStat.Stat(diff)
avg_diff = sum(stat.mean) / len(stat.mean)

# Print the number of differing pixels and the average difference
print(f"Number of differing pixels: {num_diff_pixels}")
print(f"Average difference per channel: {avg_diff:.2f}")