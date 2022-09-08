import sys 
import getopt


def inoutpath(argv):
	arg_inpath = ""
	arg_outpath = "" #arg_user = ""
	arg_help = "{0} -i <inpath> -o <outpath>".format(argv[0])
	try:
		opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "inpath=", "outpath="])
	except:
		print(arg_help)
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)  # print the help message
			sys.exit(2)
		elif opt in ("-i", "--INPATH"):
			arg_input = arg
		elif opt in ("-o", "--OUTPATH"):
			arg_output = arg


	return(arg_inpath,arg_outpath)
