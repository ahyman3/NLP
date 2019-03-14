"""
This program was adapted from the Stanford NLP class SpamLord homework assignment.
    The code has been rewritten and the data modified, nevertheless
    please do not make this code or the data public.
This version has two patterns that were suggested in comments
    in order to get you started .
"""
import sys
import os
import re
import pprint
import html

"""
TODO
For Part 1 of our assignment, add to these two lists of patterns to match
examples of obscured email addresses and phone numbers in the text.
For optional Part 3, you may need to add other lists of patterns.
"""
# email .edu patterns

# each regular expression pattern should have exactly two sets of parentheses.
#   the first parenthesis should be around the someone part
#   the second parenthesis should be around the somewhere part
#   in an email address whose standard form is someone@somewhere.edu
epatterns = []
#captures user with  letters, numbers, and periods, 0 to 3 spaces, @, 0 to 3
#spaces, @ symbol, captures domain name with letters, numbers, and periods
#must end with .edu
epatterns.append('([A-Za-z0-9.]+)\s{,3}@\s{,3}([A-Za-z0-9.]+)\.edu')
#captures username with letters, numbers, and periods, followed by an optional
#space with the < symbol and more letters and spaces and a required > and
#optional @ and optional space. Then captures the domain name of letters, numbers
#and periods, requires a .edu ending
epatterns.append('([A-Za-z0-9.]+)\s?<[A-Za-z ]+>@?\s?([A-Za-z0-9.]+)\.edu')
#Captures username of letters, numbers, and periods followed by a space and a
#special character (not letter, @, or period) followed by an optional any
#character for any length until the @ symbol, the captures the domain name of
#letters and periods, and a .edu required at the end
epatterns.append('([A-Za-z0-9.]+)\s[^A-Za-z@.].*?@([A-Za-z.]+)\.edu')

#New list for regular expression patterns
e3patterns = []
#Capitalized EDU
e3patterns.append('([A-Za-z\.]+)@([A-Za-z\.]+)(\.EDU)')
#captures email strings with an obfuscated "@" sign (at or where)
e3patterns.append('([a-z.]+) [a-z]+ ([a-z.]+)(\.edu)')
#Capturing the .com group
e3patterns.append('([a-z]+) [a-z]+ ([a-z.]+)(\.com)')

#New list of patterns
e4patterns = []
#Captures patterns that are parameters for functions, domain first then user
e4patterns.append('\'([a-z.]+.edu)\', ?\'([a-z.]+)\'')



# phone patterns
# each regular expression pattern should have exactly three sets of parentheses.
#   the first parenthesis should be around the area code part XXX
#   the second parenthesis should be around the exchange part YYY
#   the third parenthesis should be around the number part ZZZZ
#   in a phone number whose standard form is XXX-YYY-ZZZZ
ppatterns = []
#captures numbers that are formatted as 555-555-5555
ppatterns.append('(\d{3})-(\d{3})-(\d{4})')
#captures phone numbers formatted with area code first, then either a space or a
#dash. Then has the next three numbers followed by either a dash or a space
#and then four more numbers
ppatterns.append('(\d{3})[ -](\d{3})[- ](\d{4})')
#Captures phone numbers that have either a bracket or parentheses around the
#area code. Then captures the first three digits. After the area code group,
#it looks for the closing bracket of the initial bracket. Then followed by either
#a space or a dash. After the space or dash it captures the next three digits.
#Then is followed by another dash or space, and then finally captures the last
#four digits
ppatterns.append('[\(\[](\d{3})[\)\]][ -]?(\d{3})[- ](\d{4})')




"""
This function takes in a filename along with the file object and
scans its contents against regex patterns. It returns a list of
(filename, type, value) tuples where type is either an 'e' or a 'p'
for e-mail or phone, and value is the formatted phone number or e-mail.
The canonical formats are:
     (name, 'p', '###-###-#####')
     (name, 'e', 'someone@something')
If the numbers you submit are formatted differently they will not
match the gold answers

TODO
For Part 3, if you have added other lists, you should add
additional for loops that match the patterns in those lists
and produce correctly formatted results to append to the res list.
"""
def process_file(name, f):
    # note that debug info should be printed to stderr
    # sys.stderr.write('[process_file]\tprocessing file: %s\n' % (path))
    res = []

    #Pattern for domain/dot
    patDot = re.compile(" do?[mt] ")
    #Pattern for html patterns
    patHtml = re.compile('&[a-z0-9#]')
    #Html comments removal
    patComment = re.compile('<!--[^<]+--> ')
    #pattern for [a-z@.]-
    patDash = re.compile('([a-z.@]-){2,}')
    #pattern for emails that are separated by all spaces
    #Captures everython after the at
    #e.g pal at cs stanford edu
    patSpace = re.compile('at ([a-z ]+ edu)')

    for line in f:

        # you may modify the line, using something like substitution
        #    before applyting the patterns
        line = line.lower()
        #Capturing emails dash character
        if re.search(pattern=patDash, string= line) is not None:
            line = line.replace("-", "")
        #replacing the html comments
        line = re.sub(patComment, "", line)
        #pattern for dots or doms
        line = re.sub(patDot, ".", line)
        if re.search(patHtml, line) is not None:
            #if there is, make it literal in utf8
            line = html.unescape(line)
        #Finding a pattern that has all spaces after at and capturing the domain
        spaceMatch = re.findall(patSpace, line)
        #if there is a match to the pattern of all spaces
        if len(spaceMatch) != 0:
            #in the line, replace the captured domain pattern with spaces
            #with the domain pattern that has spaces replaced with "."
            line = line.replace(spaceMatch[0], spaceMatch[0].replace(" ", "."))
        #replacing ; to be a .
        #Comes after unescape to not mess up any other
        line = line.replace(";", ".")

        for epat in epatterns:
            # each epat has 2 sets of parentheses so each match will have 2 items in a list
            matches = re.findall(epat,line)
            for m in matches:
                # string formatting operator % takes elements of list m
                #   and inserts them in place of each %s in the result string
                # email has form  someone@somewhere.edu
                email = '%s@%s.edu' % m
                res.append((name,'e',email))

        #Looping through the new patterns list
        for epat in e3patterns:
            # each epat has 3 sets of parentheses so each match will have 3 items in a list
            matches = re.findall(epat,line)
            for m in matches:
                # string formatting operator % takes elements of list m
                #   and inserts them in place of each %s in the result string
                # email has form  someone@somewheresomething
                #Changed to have three different groups
                email = '%s@%s%s' % m
                res.append((name,'e',email))


        for epat in e4patterns:
            # each epat has 3 sets of parentheses so each match will have 3 items in a list
            matches = re.findall(epat,line)
            for m in matches:
                # string formatting operator % takes elements of list m
                #   and inserts them in place of each %s in the result string
                # email has form  someone@somewheresomething
                #has two strings, but places the index 1 capture group first
                #and the index 0 group second
                email = '%s@%s' % (m[1], m[0])
                res.append((name,'e',email))
        
        # phone pattern list
        for ppat in ppatterns:
            # each ppat has 3 sets of parentheses so each match will have 3 items in a list
            matches = re.findall(ppat,line)
            for m in matches:
                # phone number has form  areacode-exchange-number
                phone = '%s-%s-%s' % m
                res.append((name,'p',phone))
    return res

"""
You should not edit this function.
"""
def process_dir(data_path):
    # save complete list of candidates
    guess_list = []
    # save list of filenames
    fname_list = []

    for fname in os.listdir(data_path):
        if fname[0] == '.':
            continue
        fname_list.append(fname)
        path = os.path.join(data_path,fname)
        f = open(path,'r', encoding='latin-1')
        # get all the candidates for this file
        f_guesses = process_file(fname, f)
        guess_list.extend(f_guesses)
    return guess_list, fname_list

"""
You should not edit this function.
Given a path to a tsv file of gold e-mails and phone numbers
this function returns a list of tuples of the canonical form:
(filename, type, value)
"""
def get_gold(gold_path):
    # get gold answers
    gold_list = []
    f_gold = open(gold_path,'r', encoding='latin-1')
    for line in f_gold:
        gold_list.append(tuple(line.strip().split('\t')))
    return gold_list

"""
You should not edit this function.
Given a list of guessed contacts and gold contacts, this function
    computes the intersection and set differences, to compute the true
    positives, false positives and false negatives.
It also takes a dictionary that gives the guesses for each filename,
    which can be used for information about false positives.
Importantly, it converts all of the values to lower case before comparing.
"""
def score(guess_list, gold_list, fname_list):
    guess_list = [(fname, _type, value.lower()) for (fname, _type, value) in guess_list]
    gold_list = [(fname, _type, value.lower()) for (fname, _type, value) in gold_list]
    guess_set = set(guess_list)
    gold_set = set(gold_list)

    # for each file name, put the golds from that file in a dict
    gold_dict = {}
    for fname in fname_list:
        gold_dict[fname] = [gold for gold in gold_list if fname == gold[0]]

    tp = guess_set.intersection(gold_set)
    fp = guess_set - gold_set
    fn = gold_set - guess_set

    pp = pprint.PrettyPrinter()
    #print 'Guesses (%d): ' % len(guess_set)
    #pp.pprint(guess_set)
    #print 'Gold (%d): ' % len(gold_set)
    #pp.pprint(gold_set)

    print ('True Positives (%d): ' % len(tp))
    # print all true positives
    pp.pprint(tp)
    print ('False Positives (%d): ' % len(fp))
    # for each false positive, print it and the list of gold for debugging
    for item in fp:
        fp_name = item[0]
        pp.pprint(item)
        fp_list = gold_dict[fp_name]
        for gold in fp_list:
            s = pprint.pformat(gold)
            print('   gold: ', s)
    print ('False Negatives (%d): ' % len(fn))
    # print all false negatives
    pp.pprint(fn)
    print ('Summary: tp=%d, fp=%d, fn=%d' % (len(tp),len(fp),len(fn)))

"""
You should not edit this function.
It takes in the string path to the data directory and the gold file
"""
def main(data_path, gold_path):
    guess_list, fname_list = process_dir(data_path)
    gold_list =  get_gold(gold_path)
    score(guess_list, gold_list, fname_list)

"""
commandline interface assumes that you are in the directory containing "data" folder
It then processes each file within that data folder and extracts any
matching e-mails or phone numbers and compares them to the gold file
"""
if __name__ == '__main__':
    print ('Assuming ContactFinder.py called in directory with data folder')
    main('data/dev', 'data/devGOLD')
