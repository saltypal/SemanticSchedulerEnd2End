from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
api=KaggleApi(); api.authenticate(); token=None; page=0
while True:
    req=ApiListKernelSessionOutputRequest(); req.user_name='satyapaladuguplus'; req.kernel_slug='semantic-image-pool-scene-cache-20261030'; req.page_size=200
    if token: req.page_token=token
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.list_kernel_session_output(req)
    page+=1
    for item in response.files or []:
        if item.file_name.endswith('Mini_dataset_6000.tar') or 'Mini_dataset_6000.tar' in item.file_name:
            print('FOUND_PAGE',page,'FILE',item.file_name)
    token=response.next_page_token
    if not token:
        print('NO_MORE_PAGES',page); break
