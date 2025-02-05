
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

PDF_DIR = ../../documents/lectures/winter_2023/website_notes/planning_in_mdps

define generate_pdf
	$(CHROME) --headless --disable-gpu --print-to-pdf=lec$(1).pdf https://rltheory.github.io/lecture-notes/planning-in-mdps/lec$(1)/
	mv lec$(1).pdf $(PDF_DIR)/lec$(1).pdf
endef

all: lec1 lec2 lec3 lec4 lec5 lec6 lec7 lec8 lec9 lec10 lec11 lec12 lec13 lec14 lec15 lec16 join_pdfs

join_pdfs: $(PDF_DIR)
	pdflatex -output-directory $(PDF_DIR) $(PDF_DIR)/joint_notes.tex

lec1:
	$(call generate_pdf,1)

lec2: 
	$(call generate_pdf,2)

lec3: 
	$(call generate_pdf,3)

lec4: 
	$(call generate_pdf,4)

lec5: 
	$(call generate_pdf,5)

lec6: 
	$(call generate_pdf,6)

lec7: 
	$(call generate_pdf,7)

lec8: 
	$(call generate_pdf,8)

lec9: 
	$(call generate_pdf,9)

lec10: 
	$(call generate_pdf,10)

lec11: 
	$(call generate_pdf,11)

lec12: 
	$(call generate_pdf,12)

lec13: 
	$(call generate_pdf,13)

lec14: 
	$(call generate_pdf,14)

lec15: 
	$(call generate_pdf,15)

lec16: 
	$(call generate_pdf,16)