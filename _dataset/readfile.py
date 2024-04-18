import os

folder = '.'
folder = os.path.abspath(folder)

image_list_path = "_image_list_"

image_list_path = os.path.join(folder, image_list_path)

os.makedirs(image_list_path, exist_ok=True)

domains = os.listdir(folder)
domains.sort()


for d in range(len(domains)):
	dom = domains[d]

	if dom in ["annotation", "_image_list_", "_image_list"]:
		continue

	if os.path.isdir(os.path.join(folder, dom)):
		dom_new = dom.replace(" ","_")
		print(dom, dom_new)
		# os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))

		classes = os.listdir(os.path.join(folder, dom_new))
		classes.sort()
		print(classes)
		# f = open(dom_new[0] + "_list.txt", "w")
		# file_path = os.path.join(image_list_path, dom_new[0] + "_list.txt")
		file_path = os.path.join(image_list_path, dom_new + "_list.txt")
		f = open(file_path, "w")
		for c in range(len(classes)):
			cla = classes[c]
			cla_new = cla.replace(" ","_")
			# print(cla, cla_new)
			# os.rename(os.path.join(folder, dom_new, cla), os.path.join(folder, dom_new, cla_new))
			files = os.listdir(os.path.join(folder, dom_new, cla_new))
			files.sort()
			# print(files)
			for file in files:
				file_new = file.replace(" ","_")
				# os.rename(os.path.join(folder, dom_new, cla_new, file), os.path.join(folder, dom_new, cla_new, file_new))
				# print(file, file_new)
				# print('{:} {:}'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
				f.write('{:} {:}\n'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
		f.close()