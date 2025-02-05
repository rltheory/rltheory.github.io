
# Set the path to chrome depending on OS
CHROME := 
ifeq ($(OS),Windows_NT)
	CHROME += /windows/chrome/path 
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CHROME += /windows/chrome/path 
	endif
	ifeq ($(UNAME_S),Darwin)
		CHROME += /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome 
	endif
endif

PDF_DIR = ../../documents/lectures/winter_2022/website_notes/online_rl

define generate_pdf
	$(CHROME) --headless --disable-gpu --print-to-pdf=lec$(1).pdf https://rltheory.github.io/lecture-notes/online-rl/lec$(1)/
	mv lec$(1).pdf $(PDF_DIR)/lec$(1).pdf
endef

all: lec22 lec23 lec24 join_pdfs

join_pdfs: $(PDF_DIR)
	pdflatex -output-directory $(PDF_DIR) $(PDF_DIR)/joint_notes.tex

lec22: 
	$(call generate_pdf,22)

lec23: 
	$(call generate_pdf,23)

lec24: 
	$(call generate_pdf,24)