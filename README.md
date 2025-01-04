## ColQwen2 retrieval via infinity embedding api and local vespa database - example


1. Clone repo
2. Create python venv or conda env
3. pip install -r requirements.txt
4. Download "Vespa CLI" package for your platform here -> https://github.com/vespa-engine/vespa/releases
   - copy bin/vespa to /usr/bin/vespa
   - Adjust permissions:
   ```
   sudo chmod +x /usr/bin/vespa && sudo chmod 755 /usr/bin/vespa
   ```
6. Start vespa local docker container
   ```
   docker run --detach --name vespa --hostname vespa-container --publish 8080:8080 --publish 19071:19071 vespaengine/vespa:latest
   ```
7. Start ColQwen2 via infinity embedding api like (only -merged versions work with infinity):
   ```
   infinity_emb v2 --device cuda --no-bettertransformer --batch-size 16 --dtype float16 --model-id vidore/colqwen2-v1.0-merged --served-model-name colqwen2 --api-key sk-1234
   ```
8. Deploy vespa application schema to vespa instance
   ```
   python deploy_vespa_app_local.py --vespa_application_name MyApplicationName
   ```

9. Embed and feed pdf files from a folder to vespa
   ```
   python feed-vespa_colqwen2-api4.py --application_name MyApplicationName --vespa_schema_name pdf_page --pdf_folder /path/to/my/pdf/files/
   ```

10. Use query_vespa.py to test retrieval. Adjust "queries = [...]" in file as needed.
   ```
   python query_vespa.py
   ```
   
