mkdir ~/.chainlit
echo "[project]" > ~/.chainlit/config.toml
echo "enable_telemetry = true" >> ~/.chainlit/config.toml
echo "session_timeout = 3600" >> ~/.chainlit/config.toml
echo "[ UI]" >> ~/.chainlit/config.toml
echo "hide_cot = true" >> ~/.chainlit/config.toml
cat << EOF >> ~/.chainlit/config.toml
EOF
     
chainlit run app.py --host "0.0.0.0" --port 8888 -w