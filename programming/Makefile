all: compile checkstyle

compile:
	python3 -m py_compile *.py

checkstyle:
	flake8 *.py

clean:
	rm -rf __pycache__
	rm -f *.pyc
