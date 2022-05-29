import xlwings as xw
from docxtpl import DocxTemplate
import os, sys

os.chdir((sys.path[0]))
def main():
    wb = xw.Book.caller()
    sht_panel = wb.sheets['Panel']
    print(sht_panel)
    doc = DocxTemplate('SS_template.docx')
    print(doc)

    # -- Get Values from Excel
    print(sht_panel.range('A2'))
    context = sht_panel.range('A2').options(dict, expand='table', numbers=int).value
    print(context)

    # -- Render & save Word document
    output_name = f'Template_rendered_{context}.docx'
    doc.render(context)
    doc.save('Template_Rendered2.docx')

if __name__ == "__main__":
    xw.Book("word_automation.xlsm").set_mock_caller()
    main()
