import os, random, sys, time

def main():
    args = parse_args()
    delete_region(args.dir, args.node)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',  type=str,
                        help='The data directory, e.g., /tmp/tiny-imagenet-200')
    parser.add_argument('node', type=str,
                        help='The node, e.g., 1.2.1.2')
    args = parser.parse_args()
    return args

def extract_region(region_str, total_size):
    left = 0
    right = total_size - 1
    for i in region_str[1:]:
        if i == '.':
            continue
        middle = (left + right) // 2
        if i == '1':
            right = middle
        else:
            left = middle + 1
    return (left, right)

def delete_region(dir_name, region_str):
    file_list = []
    print("dir_name: " + dir_name)
    for root, _, files in os.walk(dir_name):
        file_list.extend(map(lambda x: os.path.join(root, x),
                             filter(lambda x: os.path.splitext(x)[1] == '.JPEG', files)))
    random.seed(112233)
    random.shuffle(file_list)
    file_count = len(file_list)
    print("Delete region: '%s'" % region_str)
    region = extract_region(region_str, file_count)
    deleted_count = region[1] - region[0] + 1
    if deleted_count == 0:
        raise ValueError("preprocess.py: No training files! " +
                         "Looked in: '%s'" % dir_name)
    pct = int(100.0 * deleted_count / file_count)
    print("Deleting region {0} amounting to {1}/{2} files ({3}%) from {4}"
          .format(region, deleted_count, file_count, pct, dir_name))
    for i in range(region[0], region[1] + 1):
        os.rename(file_list[i], file_list[i] + "-bak")
    return file_count - deleted_count
    print("Done!")

def restore_dir(dir_name):
    print("restore_dir: " + dir_name)
    start = time.time()
    file_list = []
    for root, _, files in os.walk(dir_name):
        file_list.extend(map(lambda x: os.path.join(root, x),
                         filter(lambda x: os.path.splitext(x)[1] == '.JPEG-bak', files)))
    print("restoring %i files ..." % len(file_list))

    for fn in file_list:
        os.rename(fn, fn.replace(".bak", ""))
    stop = time.time()
    print("time restore: %2.3f" % (stop-start))

# Stand-alone mode
if __name__ == "__main__":
    main()
