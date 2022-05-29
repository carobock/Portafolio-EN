from email_validator import validate_email
from unittest import skip
import pandas as pd
import numpy as np
import xlsxwriter
import re
import csv

df = pd.read_excel("Email_list.xlsx")
emails = df['Email'].tolist()
countries = df['Country'].tolist()
email_country_list = []

for country, email in zip(countries, emails):
	# print(country,email)
	if email == '' or pd.isnull(email):
		skip
	else:
		email_country_list.append((country, email))

mod_email_country_list = []
for elem in email_country_list:
	new_elem = elem[1].strip()  # Remove trailing spaces
	new_elem = re.sub('\s+', '', new_elem)  # RE for blank spaces
	mod_email_country_list.append((elem[0], new_elem))

#print(mod_email_country_list)
clean_emails=[]
for elem in mod_email_country_list:
	try:
		validate_email(elem[1])
		message = "Valid"
	except Exception:
		message = "Not Valid"
	#print(elem[1],message)
	clean_emails.append((elem[0],elem[1],message))
#print(clean_emails)

with open('./result.csv', "w", newline="", encoding='utf-8') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for row in clean_emails:
		spamwriter.writerow(row)

# workbook = xlsxwriter.Workbook('result2.xlsx')
# worksheet = workbook.add_worksheet()
#
# for row, line in enumerate(mod_email_country_list):
# #	for col, cell in enumerate(line):
# #		worksheet.write(row, col, cell)

#workbook.close()


#

