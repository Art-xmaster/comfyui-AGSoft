import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "AGSoft.LoadAudio",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftLoadAudio") return;
        const combo = node.widgets?.find(w => w.name === "audio");
        if (!combo) return;

        // ---------- Плеер (скрытый <audio> + controls) ----------
        const audioEl = document.createElement("audio");
        audioEl.controls = true;
        audioEl.style.width = "100%";
        const playerWidget = node.addDOMWidget("agsoft_audio_player", "div", audioEl,
            { serialize: false, hideOnZoom: false });
        playerWidget.computeSize = () => [200, 40];

        const updateSrc = () => {
            const file = combo.value;
            if (!file) { audioEl.removeAttribute("src"); return; }
            const filename = String(file).split(/[/\\]/).pop();
            audioEl.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
            audioEl.load();
        };

        // ---------- Кнопка загрузки ----------
        const wrap = document.createElement("div");
        const btn = document.createElement("button");
        btn.textContent = "choose file to upload";
        btn.style.width = "100%";
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "audio/*";
        fileInput.style.display = "none";
        btn.onclick = (e) => { e.preventDefault(); fileInput.click(); };

        fileInput.onchange = async () => {
            const f = fileInput.files && fileInput.files[0];
            if (!f) return;
            try {
                const body = new FormData();
                body.append("image", f, f.name);
                body.append("type", "input");
                body.append("subfolder", "");
                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                if (!resp.ok) throw new Error("upload failed: " + resp.status);
                const data = await resp.json();
                const name = data.name || f.name;
                const vals = combo.options?.values || combo.values || [];
                if (Array.isArray(vals) && !vals.includes(name)) vals.push(name);
                combo.value = name;
                if (combo.callback) combo.callback(name);
                updateSrc();
            } catch (e) {
                console.error("[AGSoft Load Audio] upload error:", e);
            }
        };

        wrap.appendChild(btn);
        wrap.appendChild(fileInput);
        const uploadWidget = node.addDOMWidget("agsoft_audio_upload", "div", wrap,
            { serialize: false, hideOnZoom: false });
        uploadWidget.computeSize = () => [200, 30];

        const oldCb = combo.callback;
        combo.callback = (v) => { if (oldCb) oldCb(v); updateSrc(); };

        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            updateSrc();
        };

        updateSrc();
        node.setSize(node.computeSize());
        app.graph.setDirtyCanvas(true);
    }
});