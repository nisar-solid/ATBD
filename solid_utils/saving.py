import os
import shutil
import pandas as pd

def save_results(save_dir:str, run_date:str, requirement:str, site:str, method:str,
                 sitedata:dict, gnss_insar_figs, validation_figs:list, validation_table:pd.DataFrame,
                 ts_functions:dict=None, summary:str=""):
    """Save the input parameters and results to an output folder.
    The results folder will include image files, and an HTML file
    recording the inputs, essential figures, and validation table.
    """
    # Create new directory for outputs
    if os.path.exists(save_dir):
        print(f"Directory {save_dir} exists")
    else:
        print(f"Creating {os.path.basename(save_dir)}")
        os.makedirs(save_dir)

    # Begin HTML string
    html_str = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{requirement:s} Method {method}</title>
</head>
<body>
    <h1>{requirement:s} requirement validation 
{site:s} site, Method {method}</h1>
</body>
""".format(requirement=requirement.title(),
           site=site,
           method=method)

    # Run date time
    html_str += """<body>
    <h2>Run time:</h2>
    {run_date:s}
    </body>
""".format(run_date=run_date)

    # Site parameters
    html_str += "<body>"
    html_str += """<h2>Setup parameters:</h2>
<ul>
"""
    for key, value in sitedata.items():
        html_str += "<li>{}: {}</li>".format(key, value)
    html_str += """</ul>
</body>"""

    # Timeseries basis functions
    if ts_functions is not None:
        html_str += "<body>"
        html_str += """<h3>Timeseries basis functions</h3>
    <ul>
    """
        for key, value in ts_functions.items():
            html_str += "<li>{}: {}</li>".format(key, value)
        html_str += """</ul>
</body>"""

    # Processing results
    html_str += """<body>
<h2>InSAR and GNSS LOS Velocities</h2>
Visual comparison of GNSS and InSAR LOS velocities
<br>
</body>
""".format(method=method)

    # Save GNSS InSAR figures
    for i, gnss_insar_fig in enumerate(gnss_insar_figs):
        fig_name = f"{requirement:s}_Method{method}_gnss_insar_figure{i + 1:d}.png"
        fig_path = os.path.join(save_dir, fig_name)
        print(f"Saving GNSS-InSAR figure to: {fig_name}")
        gnss_insar_fig.savefig(fig_path, bbox_inches='tight', transparent=True,
                               dpi=300)

        # Embed GNSS-InSAR figures
        html_str += """<body>
<img src="{fig_name:s}" alt="Method {method} GNSS InSAR Image" width="800">
</body>
""".format(fig_name=fig_name,
           method=method)

    # Validation results
    html_str += """<body>
<h2>Method {method} Results</h2>
</body>
""".format(method=method)

    # Save validation figures
    for i, validation_fig in enumerate(validation_figs):
        fig_name = f"{requirement:s}_Method{method}_validation_figure{i + 1:d}.png"
        fig_path = os.path.join(save_dir, fig_name)
        print(f"Saving validation figure to: {fig_name}")
        validation_fig.savefig(fig_path, bbox_inches='tight', transparent=True,
                               dpi=300)
    
        # Embed validation figure
        html_str += """<body>
    <img src="{fig_name:s}" alt="Method {method} Validation Image" width="800">
    </body>
    """.format(fig_name=fig_name,
               method=method)

    # Validation table
    html_str += """<body>
{}
</body>""".format(validation_table.to_html())

    # Save summary
    html_str += """<body>
<h2>Summary</h2>
{:s}
</body>""".format(summary)
    
    # Write to HTML file
    html_name = f"{requirement:s}_Method{method}_validation_report.html"
    html_path = os.path.join(save_dir, html_name)
    with open(html_path, 'w') as html_file:
        html_file.write(html_str)

    # pdf_file = html_path.split(".")[0]+".pdf"
    # HTML(html_path).write_pdf(pdf_file)
    # print(f"Saved PDF version of report to: {pdf_file}")
    print(f"Saved parameters and results to: {save_dir:s}")

    # Save to zip file
    shutil.make_archive(save_dir, 'zip', save_dir)
