def change_color():
    color_txt = r'<REDACTED>'
    label_to_rgb = {}
    with open(color_txt, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                label, hex_color = parts
                label_to_rgb[int(label)] = hex_to_rgb(hex_color)
            else:
                print(f'Error parsing line: {line}')
    return label_to_rgb


def hex_to_rgb(hex_str):
    hex_str = hex_str.replace('#', '')
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))



