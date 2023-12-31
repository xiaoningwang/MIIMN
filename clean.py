import re
import string
#import nltk
#from bs4 import BeautifulSoup

no_meanings = ['href', 'www', 'http', 'com', 'wikipedia', 'wiki', 'org', 'php', 'amp']

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])


def strip_punctuation(s):
    # print "output_2", s
    s = s.encode('raw_unicode_escape')
    return s.translate(str.maketrans("", ""), string.punctuation)
    # return ''.join(c for c in s if c not in string.punctuation)


def process_text(x):
    x = x.lower()
    x = x.replace("&quot;", " ")
    x = x.replace('"', " ")
    
    # standard clean rules
    x = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", x)
    x = re.sub(r"what's", "what is ", x)
    x = re.sub(r"\'s", " ", x)
    x = re.sub(r"\'ve", " have ", x)
    x = re.sub(r"can't", "cannot ", x)
    x = re.sub(r"n't", " not ", x)
    x = re.sub(r"i'm", "i am ", x)
    x = re.sub(r"\'re", " are ", x)
    x = re.sub(r"\'d", " would ", x)
    x = re.sub(r"\'ll", " will ", x)
    x = re.sub(r",", " ", x)
    x = re.sub(r"!", " ! ", x)
    x = re.sub(r":", " : ", x)
    x = re.sub(r"e - mail", "email", x)
    x = re.sub('[^A-Za-z0-9]+', ' ', x)  # remove non-character and non-number
    
    # newly added
    x = x.replace('copyright all rights reserved', " ")
    http_pattern = re.compile("((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")
    url_pattern = re.compile('(www|WWW)\\.[0-9%a-zA-Z\\.]+\\.(com|cn|org)')
    x = re.sub(http_pattern, " ", x)
    x = re.sub(url_pattern, " ", x)
    
    x = x.strip().split(' ')
    ans = []
    for y in x:
        if len(y) == 0 or len(y) >= 22 or y in no_meanings:
            continue
        try:  # filter number
            temp = int(y)
        except:
            ans.append(y)

    return ans


def clean_str(string, max_seq_len=-1):
    string = string.replace('"', " ")
    string = string.replace("&quot;", " ")
    string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s = string.strip().lower().split(" ")
    if max_seq_len == -1:
        return s
    elif len(s) > max_seq_len:
        return s[0:max_seq_len]
    else:
        return s


if __name__ == '__main__':
    #s = "gtfo vegetables t shirt rf1fc252993d44045a15bb1f8bf2d35e3 8na1r 540 vegetables in my diet"
    #s = "rose public gardens \\( 20070811 140220 pjg \\) here we are in the dead of winter , and by now i am sure the viewer is getting tired of the bird photos . i thought i might warm things up a bit , by digging into some shots that never got posted . if anyone knows which variety of rose this may be , i will be glad to give credit for the identification . this rose was taken at the a href http en . wikipedia . org wiki halifax public gardens rel nofollow public gardens a in halifax , nova scotia , canada in the late summer of 2007 . please view a href http bighugelabs . com flickr onblack . php \\? id 2217012047 amp bg black amp size large rel nofollow large on black a for greater detail ."
    #s = "19 10 2012 ueno park some mens were feeding birds when we arrived . one of them gave us food and taught us to feed the awesome birds \\( video soon ! \\)"
    #s = "matalasca as beach at sunset playa de matalasca as al atardecer matalasca as beach \\( huelva , spain \\) at sunset . taken handheld in available light using a lumix tz7 \\( zs3 \\) \\( 45 mm , f5 , 1 400 sec . , iso 80 \\) , this is the leftmost pic in a 3 pic panorama also available in this same set . playa de matalasca as \\( huelva \\) a la puesta de sol . tomada a pulso en luz ambiente con una lumix tz7 \\( zs3 \\) \\( 45 mm , f5 , 1 400 seg . , iso 80 \\) , esta es la foto mas a la izquierda en un panorama formado por tres fotos disponible en este mismo album ."
    #s = "abandoned cement factory \\( i \\) 33 copyright all rights reserved"
    #s = 'img 5200 , , . , . . .'
    # print(clean_str(s, 205))
    # print(clean_str(s))
    # print('\n')
    print(process_text(s))
