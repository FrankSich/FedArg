import os
import glob
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import utils
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
REPORT_MD = os.path.join(ROOT, 'RESEARCH_REPORT_FULL.md')
OUT_PDF = os.path.join(ROOT, 'RESEARCH_REPORT_FULL.pdf')

def make_equation_image(tex, path, dpi=150):
    fig = plt.figure(figsize=(6,1.2))
    fig.text(0.01, 0.5, tex, fontsize=18)
    plt.axis('off')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def get_image(path, width=400):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))

def main():
    if not os.path.exists(REPORT_MD):
        print('RESEARCH_REPORT_FULL.md not found in repo root')
        return

    # collect images under results/
    images = sorted(glob.glob(os.path.join(ROOT, 'results', '**', '*.png'), recursive=True))

    # prepare doc
    doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    if 'Code' not in styles.byName:
        styles.add(ParagraphStyle(name='Code', fontName='Courier', fontSize=8, leading=10))
    flow = []

    # Title
    flow.append(Paragraph('Mwakatobe — Full Research Report', styles['Title']))
    flow.append(Spacer(1, 12))

    # include a short table of contents placeholder
    flow.append(Paragraph('Contents: Full report, figures and code excerpts embedded.', styles['Normal']))
    flow.append(Spacer(1, 12))

    # Add the markdown content as plain paragraphs with basic heading parsing
    with open(REPORT_MD, 'r', encoding='utf-8') as f:
        md = f.read()

    in_code = False
    code_buf = []
    for line in md.splitlines():
        if line.strip().startswith('```'):
            in_code = not in_code
            if not in_code:
                # flush code block
                flow.append(Preformatted('\n'.join(code_buf), styles['Code']))
                flow.append(Spacer(1, 6))
                code_buf = []
            continue
        if in_code:
            code_buf.append(line)
            continue

        if line.startswith('# '):
            flow.append(Paragraph(line.lstrip('# ').strip(), styles['Heading1']))
            continue
        if line.startswith('## '):
            flow.append(Paragraph(line.lstrip('#').strip(), styles['Heading2']))
            continue
        if line.startswith('- '):
            flow.append(Paragraph(line.lstrip('- ').strip(), styles['Bullet']))
            continue
        if line.strip() == '':
            flow.append(Spacer(1, 6))
            continue
        # default normal
        flow.append(Paragraph(line, styles['Normal']))

    flow.append(Spacer(1, 12))

    # Add equations as images
    eq1 = os.path.join(ROOT, 'tools', 'eq_fedavg.png')
    eq2 = os.path.join(ROOT, 'tools', 'eq_dp.png')
    make_equation_image(r'$\mathbf{w}_{t+1}=\sum_{k=1}^{K} \frac{n_k}{n}\mathbf{w}_k$', eq1)
    make_equation_image(r'$\tilde{g}=g+\mathcal{N}(0,\sigma^2 C^2)$', eq2)
    flow.append(Paragraph('Mathematical definitions', styles['Heading2']))
    flow.append(Spacer(1,6))
    flow.append(get_image(eq1, width=420))
    flow.append(Spacer(1,6))
    flow.append(get_image(eq2, width=420))
    flow.append(Spacer(1,12))

    # Include code excerpts: anonymisation and DP helpers
    flow.append(Paragraph('Code excerpts', styles['Heading2']))
    flow.append(Spacer(1,6))
    files_to_include = [
        os.path.join(ROOT, 'client', 'data_utils.py'),
        os.path.join(ROOT, 'client', 'client_app.py'),
        os.path.join(ROOT, 'server', 'server_app.py')
    ]
    for p in files_to_include:
        if os.path.exists(p):
            flow.append(Paragraph(os.path.basename(p), styles['Heading3']))
            try:
                txt = open(p, 'r', encoding='utf-8').read()
                # limit size
                if len(txt) > 20000:
                    txt = txt[:20000] + '\n\n# (truncated)'
                flow.append(Preformatted(txt, styles['Code']))
                flow.append(Spacer(1, 12))
            except Exception as e:
                flow.append(Paragraph(f'Could not read {p}: {e}', styles['Normal']))

    # Add all result PNGs as a figures section
    flow.append(Paragraph('Figures (all PNGs from results/)', styles['Heading2']))
    flow.append(Spacer(1,6))
    for img in images:
        try:
            flow.append(Paragraph(os.path.relpath(img, ROOT), styles['Normal']))
            flow.append(get_image(img, width=420))
            flow.append(Spacer(1,8))
        except Exception as e:
            flow.append(Paragraph(f'Failed to include image {img}: {e}', styles['Normal']))

    # finalize
    doc.build(flow)
    print('PDF written to', OUT_PDF)

if __name__ == '__main__':
    main()
