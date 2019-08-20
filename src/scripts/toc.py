import sys
import re


line_len = 78
sub_len = 9
fig_len = 22


def spaced(line, pg, length = line_len, sp = " "):
    for i in range(int((length - len(line))/(4 if sp=='\t' else 1))):
        line = line + sp
    line = line + pg
    return line


def save(lines, filename):
    f = open(filename, "w", encoding="latin1")
    f.writelines([l + "\n" for l in lines])
    f.close()
    print("save output in " + str(filename))


def add_toc(filename):
    print("add_toc.filename = " + str(filename))
    f = open(filename, "r") 
    (ptoc, pfigs, toc, figs) = ([], [], [], [])
    for l in f.readlines():
        l = l.strip()
        if re.match("^[0-9]+$", l):
            toc.extend([spaced(t, l) for t in ptoc])   
            figs.extend([spaced(f, l) for f in pfigs])
            (pfigs, ptoc) = ([], [])
        if re.match("^(\s*Chapter\s+[0-9\.]+:.*|Abstract|Acknowledgements|List Of Figures|List Of Appendices|Appendices|Appendix|References)$", l):
            ptoc.append(l)
        if re.match("^\s*[0-9A-Z]\.[0-9\.]+\s.+$", l):
            lnum = re.split("\s+", l)[0]
            heading = re.sub(lnum, "", l).strip()
            ptoc.append("      " + spaced(lnum, "", sub_len, " ") + heading)
        if re.match("^\s*(Figure|Illustration|Appendix)\s+[A-Z0-9\.]+:.*$", l):
            lnum = re.split(":", l)[0] + ":"
            heading = re.sub(lnum, "", l).strip()
            pfigs.append(spaced(lnum, "", fig_len, " ") + heading)
    f.close()
    save(toc + ["",""] + figs, re.sub("\.", "_toc.", filename))


if __name__ == "__main__":
    add_toc(sys.argv[1])
