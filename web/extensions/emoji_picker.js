import { app } from "../../../scripts/app.js";
import { mkName } from "../utils.js";

const PACKAGE_NAME = "IrodoriTTS";
const CLASS_NAMES = [
    mkName(PACKAGE_NAME, "EmojiPicker"), 
];

const EMOJI_LIST = [
    ["👂", "囁き、耳元の音"], 
    ["😮‍💨", "吐息、溜息、寝息"], 
    ["⏸️", "間、沈黙"], 
    ["🤭", "笑い、くすくす、含み笑い"], 
    ["🥵", "喘ぎ、うめき声、唸り声"], 
    ["📢", "エコー、リバーブ"], 
    ["😏", "からかうように、甘えるように"], 
    ["🥺", "声を震わせながら、自信のなさげに"], 
    ["🌬️", "息切れ、荒い息遣い、呼吸音"], 
    ["😮", "息をのむ"], 
    ["👅", "舐める音、咀嚼音、水音"], 
    ["💋", "リップノイズ"], 
    ["🫶", "優しく"], 
    ["😭", "嗚咽、泣き声、悲しみ"], 
    ["😱", "悲鳴、叫び、絶叫"], 
    ["😪", "眠そうに、気だるげに"], 
    ["😴", "寝言、いびき"], 
    ["⏩", "早口、一気にまくしたてる、急いで"], 
    ["📞", "電話越し、スピーカー越しのような音"], 
    ["🐢", "ゆっくりと"], 
    ["🥤", "唾を飲み込む音"], 
    ["🤧", "咳き込み、鼻をすする、くしゃみ、咳払い"], 
    ["😒", "舌打ち"], 
    ["😰", "慌てて、動揺、緊張、どもり"], 
    ["😆", "喜びながら"], 
    ["💥", "勢いよく、勢いに任せて"], 
    ["😠", "怒り、不満げに、拗ねながら"], 
    ["😲", "驚き、感嘆"], 
    ["🥱", "あくび"], 
    ["😖", "苦しげに"], 
    ["😟", "心配そうに"], 
    ["🫣", "恥ずかしそうに、照れながら"], 
    ["🙄", "呆れたように"], 
    ["😊", "楽しげに、嬉しそうに"], 
    ["😎", "得意げに、自信ありげに"], 
    ["👌", "相槌、頷く音"], 
    ["🙏", "懇願するように"], 
    ["🥴", "酔っ払って"], 
    ["🎵", "鼻歌"], 
    ["🤐", "口を塞がれて"], 
    ["😌", "安堵、満足げに"], 
    ["🤔", "疑問の声"], 
    ["💪", "力を込めて、力強く"], 
    ["👃", "匂いを嗅ぐ音"], 
    ["📖", "ナレーション、独白、モノローグ"], 
];

const extension = {
    name: mkName(PACKAGE_NAME, "EmojiPicker"), 

    beforeRegisterNodeDef: async function(nodeType, nodeData, app) {
        if (!CLASS_NAMES.includes(nodeType.comfyClass)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const res = onNodeCreated?.apply(this, arguments);

            EMOJI_LIST.forEach(((emojiMap) => {
                const emoji = emojiMap[0];
                const desc = emojiMap[1];
                const buttonLabel = emoji + ": " + desc;
                this.addProperty(buttonLabel, true, "boolean", {
                    callback: (_, value) => {
                        this._updateButtonVisible(buttonLabel, value);
                    }
                });
                this.addWidget("button", buttonLabel, "emoji", () => {
                    if (navigator.clipboard && window.isSecureContext) {
                        navigator.clipboard.writeText(emoji);
                    } else {
                        const ta = document.createElement("textarea");
                        ta.value = emoji;
                        ta.style.position = "fixed";
                        ta.style.opacity = "0";
                        document.body.appendChild(ta);
                        ta.select();
                        document.execCommand("copy");
                        document.body.removeChild(ta);
                    }
                    app.extensionManager.toast.add({
                        severity: "info",
                        summary: "Information",
                        detail: emoji+"をコピーしました",
                        life: 1000
                    });
                });
            }));

            return res;
        };

        nodeType.prototype._updateButtonVisible = function(label, visible) {
            const button = this.widgets.find(w => w.name == label);
            if (button) {
                button.hidden = !visible
            }
        };

        nodeType.prototype._syncVisible = function() {
            for (const [key, value] of Object.entries(this.properties)) {
                const button = this.widgets.find(w => w.name == key);
                if (button) {
                    button.hidden = !value
                }
            }
        };
        

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function() {
            const res = configure?.apply(this, arguments);
            this._syncVisible()
            return res;
        };
    }
};

app.registerExtension(extension);
