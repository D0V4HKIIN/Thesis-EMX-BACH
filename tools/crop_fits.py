import color_print
import pathlib
import sys
from astropy.io import fits

def main(args):
    if len(args) != 4:
        print(f"Usage: {pathlib.Path(__file__).name} <input> <output> <width> <height>")
        return False
    
    color_print.init()
    
    input_path = pathlib.Path(args[0])

    if not input_path.exists():
        print(f"{color_print.RED}Input does not exist!")
        return False
    
    output_path = pathlib.Path(args[1])

    # Parse arguments
    try:
        new_width = int(args[2])
    except:
        print(f"{color_print.RED}Invalid width specified!")
        return False
    
    try:
        new_height = int(args[3])
    except:
        print(f"{color_print.RED}Invalid height specified!")
        return False
    
    # Load
    print(f"Loading image from '{input_path.resolve()}'...")
    file = fits.open(input_path)

    old_height = len(file[0].data)
    old_width = len(file[0].data[0])
    
    if new_width > old_width or new_height > old_height:
        print(f"{color_print.RED}Invalid dimensions specified. Image is {old_width}x{old_height} while new dimensions specified are {new_width}x{new_height}")
        return False
    
    # Crop
    print(f"Cropping to {new_width}x{new_height}...")

    for i in range(len(file)):
        file[i].data = file[i].data[0:new_height,0:new_width]

    # Save
        print(f"Saving output to '{output_path.resolve()}'...")
    file.writeto(output_path, overwrite=True)

    return True

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
