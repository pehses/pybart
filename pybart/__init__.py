from _pybart import call_bart #, bart

import numpy as np

def random_name(N=8):
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=N))


def bart(nargout, cmd, *args, **kwargs):

    input_data = {}
    bart_cmd = ['bart']  # we need to add 'bart' for new versions of bart
    cmd = cmd.strip()
    if len(cmd) > 0:
        bart_cmd[1:] = cmd.split(' ')
    for key, item in (*kwargs.items(), *zip([None]*len(args), args)):
        if key is not None:
            kw = ("--" if len(key) > 1 else "-") + key
            bart_cmd.append(kw)
        name = random_name() + '.mem'
        bart_cmd.append(name)
        if item.dtype != np.complex64:
            item = item.astype(np.complex64)
        item = np.asfortranarray(item)
        input_data[name.encode('utf-8')] = item

    outfiles = []
    for _ in range(nargout):
        # create memcfl names for output
        name = random_name() + '.mem'
        outfiles.append(name.encode('utf-8'))
        bart_cmd.append(name)

    output, errcode, stdout = call_bart(bart_cmd, input_data, outfiles)

    bart.ERR, bart.stdout, bart.stderr = errcode, stdout, None

    import logging
    # if it can be assumed that command is directly executed by user, print to stdout
    print_to_stdout = logging.getLogger().getEffectiveLevel() >= logging.WARNING

    # print to stdout
    for stdout_line in stdout:  # iter(stdout.readline(), ""):
        stdout += stdout_line
        if print_to_stdout:
            print(stdout_line, end="")
        logging.info(stdout_line[:-1])  # remove '/n' at the end

    if errcode:
        print(f"Command exited with error code {errcode}.")
        return

    if nargout == 0:
        return
    elif nargout == 1:
        return output[0]
    else:
        return output
