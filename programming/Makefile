all: compile checkstyle

compile:
	python3 -m py_compile *.py

checkstyle:
	flake8 *.py

lint:
	autopep8 --in-place --aggressive *.py

clean:
	rm -rf __pycache__
	rm -f *.pyc
