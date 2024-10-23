libraries = [("Library1", "1.0"), ("Library2", "2.1"), ("Library3", "3.2")]

for lib, ver in libraries:
    print("{:<15} {:<5} {:<5}".format(lib, ":", ver))