def split(content):
    cur_line = 0
    size = len(content)
    with open('input_train.txt', 'a') as f:
        while cur_line < int(size * 0.8):
            f.write(content[cur_line])
            cur_line += 1
    with open('input_test.txt', 'a') as f1:
        while cur_line < size:
            f1.write(content[cur_line])
            cur_line += 1
    f.close()
    f1.close()




if __name__ == '__main__':
    with open("input.txt") as f:
        content = f.readlines()
    f.close()
    split(content)
