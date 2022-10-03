import dropbox
import os
from tqdm import tqdm

def upload(
        access_token,
        file_path,
        target_path,
        timeout=900,
        chunk_size=4 * 1024 * 1024,
):
    dbx = dropbox.Dropbox(access_token, timeout=timeout)
    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)
        if file_size <= chunk_size:
            dbx.files_upload(f.read(), target_path, mode=dropbox.files.WriteMode(u'add', None), autorename=True)
        else:
            with tqdm(total=file_size, desc="Uploaded") as pbar:
                upload_session_start_result = dbx.files_upload_session_start(
                    f.read(chunk_size)
                )
                pbar.update(chunk_size)
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell(),
                )
                commit = dropbox.files.CommitInfo(path=target_path)
                while f.tell() < file_size:
                    if (file_size - f.tell()) <= chunk_size:
                        print(
                            dbx.files_upload_session_finish(
                                f.read(chunk_size), cursor, commit
                            )
                        )
                    else:
                        dbx.files_upload_session_append(
                            f.read(chunk_size),
                            cursor.session_id,
                            cursor.offset,
                        )
                        cursor.offset = f.tell()
                    pbar.update(chunk_size)

# path=root_dir= folder that you want to upload
def upload_folder(root_dir, path, dropbox_path, WBS_DROPBOX_ACCESS_TOKEN):
    
    for root, dirs, files in os.walk(path):
        if dirs:
            for directory in dirs:
                upload_folder(root_dir, path + '/' + directory, dropbox_path + '/' + directory, WBS_DROPBOX_ACCESS_TOKEN)
        else:
            for filename in files:
                path_to_upload = path + '/' + filename
                dropbox_file_path = dropbox_path + path_to_upload.replace(root_dir,'')
                dropbox_file_path = '/' + path_to_upload
                upload(WBS_DROPBOX_ACCESS_TOKEN, path_to_upload, dropbox_file_path)
        break