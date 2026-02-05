all: compile checkstyle test

compile:
	python3 -m py_compile *.py

checkstyle:
	flake8 *.py

test:
	python3 -m doctest merge_sort.py

clean:
	rm -rf __pycache__
	rm -f *.pyc
