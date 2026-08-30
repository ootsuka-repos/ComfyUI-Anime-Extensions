# ComfyUI-Extensions

ComfyUI向けのカスタムノード集です。Irodori-TTSの音声生成と、Kanomemoの画像解析・切り抜き機能を提供します。

## 提供するノード

- Irodori-TTS: モデル読み込み、通常・Character Voice音声生成、参照音声、LoRA、各種生成設定、音声保存
- Kanomemo: imgutils解析、WD14ヒートマップ、MobileSAMマスク、キャラクター切り抜き、RGBA保存

## 必要環境

- Python 3.10以降
- `comfy_api.latest` Extension APIに対応したComfyUI
- Irodori-TTS v4.1-Smallを使う場合はPyTorch 2.10以降
- 依存パッケージの取得にGitとインターネット接続

PyTorchとtorchaudioはComfyUI側の環境を使用します。`requirements.txt`からは意図的に除外しています。

## インストール

PowerShellでComfyUIのディレクトリへ移動し、次を実行します。

```powershell
Set-Location -LiteralPath "C:\path\to\ComfyUI\custom_nodes"
git clone https://github.com/ootsuka-repos/ComfyUI-Extensions.git

Set-Location -LiteralPath "C:\path\to\ComfyUI"
.\python_embeded\python.exe -m pip install -r ".\custom_nodes\ComfyUI-Extensions\requirements.txt"
```

仮想環境版では、最後のコマンドをその環境の`python -m pip`に置き換えてください。インストール後はComfyUIを再起動します。

## モデル配置

Irodori-TTSまたはCharacter Voiceのチェックポイントは、ComfyUIの`models/checkpoints`以下へ配置します。LoRAは`models/loras`以下へ配置します。

```text
ComfyUI/
└─ models/
   ├─ checkpoints/
   │  └─ irodori_tts/
   │     └─ model.safetensors
   ├─ loras/
   │  └─ irodori_tts/
   │     └─ adapter.safetensors
   └─ irodori/
      ├─ codecs/
      ├─ tokenizers/
      └─ image_encoders/
```

tokenizer、codec、Character Voice画像エンコーダーは初回利用時に自動取得され、`ComfyUI/models/irodori`へ保存されます。Kanomemo系モデルは`ComfyUI/models/huggingface`へ保存されます。モデル本体はこのリポジトリに含まれません。

## 基本的な使い方

通常の音声生成は、`IrodoriTTS Model Loader`の`irodori_model_config`を`IrodoriTTS Sampler`へ接続します。生成された`AUDIO`はComfyUIの音声プレビュー、または`IrodoriTTS Save Audio`へ接続できます。

Character Voice対応チェックポイントでは`Irodori Character Voice Sampler`を使用し、任意でキャラクター画像を条件として接続します。

## トラブルシューティング

- ノードが表示されない: ComfyUIの起動ログにあるimportエラーを確認し、ComfyUIが使うPython環境へ依存関係をインストールしてください。
- チェックポイントが表示されない: `models/checkpoints`以下に配置し、モデル一覧を更新するかComfyUIを再起動してください。
- 初回実行が長い: Hugging Faceからモデルを取得しています。コンソールの進捗とネットワーク接続を確認してください。
- VRAM不足: codecをCPU/fp32、batch sizeを1、decode modeをsequential、runtime cache policyをoffloadまたはunloadに設定してください。

## ライセンス

コードは[MIT License](LICENSE)で提供します。ダウンロードされるモデルと各依存ライブラリには、それぞれのライセンスが適用されます。
