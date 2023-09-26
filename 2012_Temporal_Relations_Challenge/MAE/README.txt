README.txt, Created September 29, 2011 for MAE version 0.9.


Updates from MAE v0.8.4:

- DTDs can now specify default values for attributes
- It is no longer necessary to have <TEXT></TEXT> tags surrounding files that 
have not previously been annotated--now you can load a file containing only 
text, and the entire contents of the file will be made available for 
annotation.  If you only want to make a portion of a file available for 
annotating, however, you may still use files with <TEXT></TEXT> tags.
- XML output files are now well-formed, and information about encoding 
has been added to the top of the file. Files output by MAE can now be loaded 
into XML parsers such as Python's ElementTree without raising errors.

   NOTE: In order to make these files compliant with XML parsers, the offsets 
of all the tags had to be shifted up by 1 in order to account for the newline
at the end of the line containing the <TEXT> tag.  This means that annotation 
files created by versions of MAE prior to version 0.9 will be off by 1 unless
fixed.  A Python script to make the required adjustments to the files has 
been provided in this package (fix_xml.py).
