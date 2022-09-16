import pdfrw
from pdf_annotate import PdfAnnotator, Location, Appearance
from pdfrw import PdfWriter,PdfReader,PageMerge

class flow_Manipulator(object):
    def __init__(self):
        pass

    def merge(self,outfile, pdfs):
            writer = pdfrw.PdfWriter()
            for pdffile in pdfs:
                pages = pdfrw.PdfReader(pdffile).pages
                writer.addpages(pages)
            writer.write(outfile)

    ACTION_TABLE = {
    "rotate":"rotate",
    "add_annotation":"add_annotation",
    'addblankpage': 'addblankpage',
    "addwatermark":"addwatermark",
    'concatebenign': 'concatebenign',
    }

    def rotate(self,sample):
        rotate = 90
        assert rotate % 90 == 0

        try:
            trailer = PdfReader(sample)
            pages = trailer.pages

            for onerange in [[1, len(pages)]]:
                onerange = (onerange + onerange[-1:])[:2]
                for pagenum in range(onerange[0] - 1, onerange[1]):
                    pages[pagenum].Rotate = (int(pages[pagenum].inheritable.Rotate or
                                                 0) + rotate) % 360

            outdata = PdfWriter(sample)
            outdata.trailer = trailer
            outdata.write()

        except AttributeError:                          #see the details
            pass
        except pdfrw.errors.PdfParseError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass
        finally:
            pass
        return sample

    def add_annotation(self,sample):
        a = PdfAnnotator(sample)
        try:
            a.add_annotation('square',  Location(x1=50, y1=50, x2=100, y2=100, page=0),Appearance(stroke_color=(1, 0, 0), stroke_width=5))
            a.write(sample,overwrite=True)  # or use overwrite=True if you feel lucky
        except TypeError:
            pass
        except pdfrw.errors.PdfParseError:
            pass
        except ValueError:
            pass
        except AttributeError:
            pass
        finally:
            pass
        return sample

    def addblankpage(self,sample):
        try:
            self.merge(sample, [sample, r"D:\Download\DRL_attack_defense\blank.pdf"])
        except AttributeError:
            pass
        except pdfrw.errors.PdfParseError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass
        finally:
            pass
        return sample

    def addwatermark(self,sample):
        wmark = PageMerge().add(PdfReader(r"D:\Download\DRL_attack_defense\blank_with_water.pdf").pages[0])[0]
        try:
            trailer = PdfReader(sample)
            for page in trailer.pages:
                PageMerge(page).add(wmark).render()
            PdfWriter(sample, trailer=trailer).write()
        except NameError:
            pass
        except AttributeError:
            pass
        except pdfrw.errors.PdfParseError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass
        finally:
            pass
        return sample

    def concatebenign(self,sample):
        writer = PdfWriter()
        try:
            for inpfn in [sample, r"D:\Download\DRL_attack_defense\concatebenign.pdf"]:
                writer.addpages(PdfReader(inpfn).pages)
            writer.write(sample)
        except AttributeError:
            pass
        except pdfrw.errors.PdfParseError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass
        finally:
            pass
        return sample

    def modify(self,sample, _action):
        action_func=flow_Manipulator().__getattribute__(_action)
        return action_func(sample)

if __name__ == '__main__':

    k = r"D:\Download\pdf_data\123.pdf"
    a = flow_Manipulator().modify(k,"concatebenign")

