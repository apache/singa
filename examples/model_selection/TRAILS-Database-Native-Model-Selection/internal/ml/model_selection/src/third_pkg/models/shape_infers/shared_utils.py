def parse_channel_info(xstring):
    blocks = xstring.split(" ")
    blocks = [x.split("-") for x in blocks]
    blocks = [[int(_) for _ in x] for x in blocks]
    return blocks
