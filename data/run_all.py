from importlib import import_module
from tqdm import tqdm
import email
import email.mime.text
import email.mime.multipart
from smtplib import SMTP

BODY = """
Done curating all atom data. See /projects/ml/RF2_allatom/sm_compl_20240412/sm_compl.pkl for the final dataset.

// \\          // \\  // \\          // \\  // \\          // \\  // \\
\\   \\      // | :,\\': | \\      // | :,\\': | \\      // | :,\\': | \\
 \\  | |\\  //  | | // \\  | |\\  //  | |//  \\  | \\  // | | //  \\ | |
  \\ | :,\\': | //      \\ | :,\\': | //      \\ | :,\\': | //      \\ |
    \\ //  \\ //          \\ //  \\ //          \\ //  \\ //          \\
"""

def send_email(
    from_email: str = "psturm@uw.edu",
    to_email: str = "rohith@uw.edu",
    subject: str = "RosettaFold All Atom Data Curation Job Completed",
    body: str = " ",
    server: str = "mail.ipd",
):
    message = email.mime.multipart.MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(email.mime.text.MIMEText(body, "plain"))

    with SMTP(server) as conn:
        conn.send_message(message)


def main(should_send_email: bool = True):
    step_1 = import_module("1_parse_cifs_into_rows")
    step_2 = import_module("2_merge_rows_into_master_set")
    step_3 = import_module("3_filter_nonexisting_partner_chains")
    step_4 = import_module("4_load_all_examples")
    step_5 = import_module("5_add_bad_examples_as_column")

    step_1_jobs = step_1.main()
    for job in tqdm(step_1_jobs, desc="Running cif parsing..."):
        _ = job.result()

    print("Merging rows into master set...")
    step_2.main()

    print("Filtering nonexisting partner chains...")
    step_3.main()

    step_4_jobs = step_4.main()
    for job in tqdm(step_4_jobs, desc="Running example loading..."):
        _ = job.result()

    print("Adding bad examples as column...")
    step_5.main()

    if should_send_email:
        send_email()


if __name__ == "__main__":
    main()
