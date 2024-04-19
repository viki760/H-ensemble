import os

CLASS_PER_TASK = -1
TASK_PER_DOMAIN = 3

# root_path = ""

image_list_path = "./_image_list_/"
target_image_list_path = "./_image_list/"

os.makedirs(target_image_list_path, exist_ok=True)


for file in os.listdir(image_list_path):
    # continue
    # if file.startswith("caltech"):
    #     pass
    if not file.endswith("_list.txt"):
        continue
    domain = os.path.basename(file).replace("_list.txt", "").lower().replace("_", "")
    print(domain)
    file_path = os.path.join(image_list_path, file)
    with open(file_path, "r") as f:
        data_list = []
        for line in f.readlines():
            split_line = line.split()
            target = split_line[-1]
            path = ' '.join(split_line[:-1])
            if not os.path.isabs(path):
                raise NotImplementedError
            target = int(target)
            data_list.append((path, target))
    
    num_class = max(data_list, key=lambda x: x[1])[1] + 1
    # num_task = num_class // CLASS_PER_TASK
    # for i in range(num_task):
    #     with open(os.path.join(target_image_list_path, f"{domain}_{i}_list.txt", "w"))  as f:
    #         for j in range(CLASS_PER_TASK):
    #             target = j + i * CLASS_PER_TASK
    #             for d in data_list:
    #                 if d[1] == target:
    #                     f.write(f"{d[0]} {j}\n")

    num_task = TASK_PER_DOMAIN
    CLASS_PER_TASK = num_class // num_task
    print(f"{CLASS_PER_TASK} / {num_class}")
    for i in range(num_task):
        with open(os.path.join(target_image_list_path, f"{domain}_{i}_list.txt"), "w")  as f:
            for j in range(CLASS_PER_TASK):
                target = j + i * CLASS_PER_TASK
                for d in data_list:
                    if d[1] == target:
                        f.write(f"{d[0]} {j}\n")

task_list = []
for file in os.listdir(target_image_list_path):
    task_name = os.path.basename(file).replace("_list.txt", "")
    task_list.append(task_name)
task_list.sort()
print(task_list)
print(len(task_list))