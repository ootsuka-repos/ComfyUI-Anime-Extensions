# ComfyUI-Extensions

ComfyUI向けのカスタムノード集です。Irodori-TTSの音声生成、画像を条件にしたCharacter Voice音声生成、Kanomemoの画像解析・切り抜き・ぼかし処理を提供します。

## 提供するノード

### Irodori-TTS

カテゴリは`ComfyUIExtensions/IrodoriTTS`です。Character Voice Samplerはその下の`Character Voice`にあります。

| ノード名 | 機能 |
| --- | --- |
| IrodoriTTS Model Loader | チェックポイントを選び、モデル・codecのデバイス、精度、キャッシュ方針を設定 |
| IrodoriTTS Sampler | テキストから音声生成。参照音声・VoiceDesign・各種設定を任意で接続 |
| Irodori Character Voice Sampler | Character Voice対応チェックポイントで音声生成。キャラクター画像を任意で接続 |
| IrodoriTTS Reference Audio | 音声・動画ファイルを話者参照に指定。音量正規化と参照時間の上限を設定 |
| IrodoriTTS VoiceDesign Config | VoiceDesign対応モデル向けの声の説明文（`caption`）を設定 |
| IrodoriTTS CFG Config | テキスト・話者・caption・キャラクター条件のガイダンスを設定 |
| IrodoriTTS Rescale Config | rescale、truncation、speaker K/V補正を設定 |
| IrodoriTTS Schedule Config | 通常TTSのサンプリングスケジュール（`linear` / `sway`）を設定 |
| IrodoriTTS Trim Tail Config | 末尾の無音・平坦部分を切り詰める判定を設定 |
| IrodoriTTS LoRA Stack | LoRA設定を作成。現在の接続上の制約は後述 |
| IrodoriTTS Save Audio | `AUDIO`をWAV・MP3・FLACで保存 |
| IrodoriTTS Emoji Picker | 演技指定用の絵文字をクリップボードへコピーする補助UI。入出力端子なし |

### Kanomemo

カテゴリは`Kanomemo/analysis`と`Kanomemo/portrait`です。

| ノード名 | 機能・出力 |
| --- | --- |
| Kanomemo Image Analysis (imgutils) | `face` / `head` / `censor` / `nudenet` / `wd14` / `ocr`の解析結果をJSON文字列で出力 |
| Kanomemo WD14 ViT Scores | WD14 ViTのタグスコアをJSON文字列で出力。`tags`はカンマ区切り、空欄なら全タグ |
| Kanomemo Heatmap Censor (WD14 ViT) | タグのヒートマップに基づくぼかし・ピクセル化。処理済み`IMAGE`とラベルのJSON文字列を出力 |
| Kanomemo Object Mask (MobileSAM) | 矩形座標`x0, y0, x1, y1`で指定した対象の`MASK`を生成 |
| Kanomemo Character Segment (imgutils) | ISNetISでキャラクターを切り抜き、`IMAGE`と`foreground_mask`を出力。`scale`は1024固定 |
| Kanomemo Save RGBA | `IMAGE`と`foreground_mask`を透過PNGとして保存 |

解析・WD14スコア・ヒートマップ・MobileSAMの入力は1枚のRGB画像です。Character SegmentとSave RGBAは画像バッチに対応します。

## 必要環境

- `comfy_api.latest` Extension APIに対応したComfyUI
- ComfyUIと[requirements.txt](requirements.txt)の依存パッケージに対応したPython環境
- Irodori-TTS v4.1-Smallを使う場合はPyTorch 2.10以降（`requirements.txt`に記載の要件）
- 依存パッケージとモデルの取得にGitとインターネット接続

PyTorchとtorchaudioはComfyUI側の環境を使用します。`requirements.txt`からは意図的に除外しています。音声処理にはtorchcodec、モデル処理にはtransformers・DACVAE、画像処理にはdghs-imgutils・timm・ultralyticsなどを使用します。

## インストール

### Linux・仮想環境版

ComfyUIが使用する仮想環境を有効にして実行します。パスは環境に合わせて変更してください。

```bash
cd /path/to/ComfyUI
git clone https://github.com/ootsuka-repos/ComfyUI-Extensions.git custom_nodes/ComfyUI-Extensions
python -m pip install -r custom_nodes/ComfyUI-Extensions/requirements.txt
```

### Windowsポータブル版（PowerShell）

`ComfyUI`と`python_embeded`が並ぶポータブル版のルートから実行します。

```powershell
Set-Location -LiteralPath "C:\path\to\ComfyUI_windows_portable"
git clone https://github.com/ootsuka-repos/ComfyUI-Extensions.git .\ComfyUI\custom_nodes\ComfyUI-Extensions
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-Extensions\requirements.txt
```

インストール後はComfyUIを再起動します。更新時はこのリポジトリで`git pull`を実行し、同じPython環境で依存関係を再インストールして再起動してください。

## モデル配置・自動取得

Irodori-TTSまたはCharacter Voiceのチェックポイントは、ComfyUIの`models/checkpoints`以下へ配置します。モデル本体はこのリポジトリに含まれません。

```text
ComfyUI/
└─ models/
   ├─ checkpoints/
   │  └─ irodori_tts/
   │     └─ model.safetensors
   ├─ irodori/                 # Character Voice用の自動取得先
   │  ├─ codecs/
   │  ├─ tokenizers/
   │  └─ image_encoders/
   └─ huggingface/
      └─ hub/                 # Kanomemo用の自動取得先
```

- Model Loaderはチェックポイント設定の`latent_dim`からcodecを選択します。32次元は`Aratako/Semantic-DACVAE-Japanese-32dim`、128次元は`facebook/dacvae-watermarked`です。それ以外はエラーになります。
- 通常TTSのtokenizer・codecは、ローカルに存在しなければ必要に応じて取得されます。通常TTSでは`models/irodori`を保存先に指定しておらず、各ライブラリのキャッシュ設定に従います。
- Character Voiceのtokenizer・codec・画像エンコーダーは`models/irodori`以下へ取得します。
- Kanomemoは実行時にHugging Faceのキャッシュを`models/huggingface/hub`へ設定します。WD14 ViTには`SmilingWolf/wd-vit-tagger-v3`、MobileSAMには`dhkim2810/MobileSAM`を使用し、imgutils用のモデルも必要に応じて取得します。このキャッシュ設定は同じComfyUIプロセス内で共有されます。

## 基本的な使い方

### 通常TTS・VoiceDesign

1. `IrodoriTTS Model Loader`で対応チェックポイントを選び、`irodori_model_config`を`IrodoriTTS Sampler`へ接続します。
2. Samplerの`text`に読み上げる文章を入力します。必要に応じてReference Audioの出力を`ref_config`へ、VoiceDesign Configの出力を`voice_design_config`へ接続します。VoiceDesignにはcaption条件に対応したモデルが必要です。
3. Samplerの`audio`を音声プレビュー、または`IrodoriTTS Save Audio`へ接続して実行します。保存先の既定プレフィックスは`output/audio/IrodoriTTS`です。

通常TTSはduration predictorを持つチェックポイントで音声長を自動推定します。predictorがない場合は30秒にフォールバックし、`duration_scale`は使用しません。`duration_scale`（既定1.0、範囲0.1〜3.0）は推定長に掛ける倍率で、生成長は0.5〜30秒の範囲に制限されます。手動秒数を指定するUIはありません。`trim_tail`は既定で有効なので、保存される音声は末尾処理によって短くなることがあります。

Reference AudioはComfyUIの`input`内の音声・動画を選択でき、音声のアップロードにも対応します。`max_ref_seconds`は1〜120秒、既定120秒です。動画の音声抽出にはimageio-ffmpegまたはffmpegを使用します。

### Character Voice

同じModel LoaderでCharacter Voice対応チェックポイントを選び、`Irodori Character Voice Sampler`へ接続します。`character_image`は任意で、画像バッチの場合は先頭の1枚を使用します。

こちらは`seconds`で生成長を指定します（既定30秒、範囲1〜120秒）。CFG・Rescale・Trim Tailの設定を接続できますが、参照音声・VoiceDesign・LoRA・Scheduleの入力端子はありません。CFGの話者・caption設定と、Rescaleのspeaker K/V補正も使用しません。

### キャラクター切り抜き・透過PNG保存

`Kanomemo Character Segment (imgutils)`の`image`と`foreground_mask`を、`Kanomemo Save RGBA`の同名入力へ接続します。マスクは1が不透明、0が透明です。保存先の既定プレフィックスは`output/kanomemo/portrait`です。

MobileSAMで対象を選ぶ場合は、元画像上のピクセル座標で矩形を指定します。無効な矩形やマスクが得られない場合は、全て0のマスクを返します。

## 現在の制約

- **LoRAのUIと通常TTSランタイムの形式が一致していません。** LoRA Stackは`models/loras`のファイル一覧からパスを渡しますが、ランタイムは`adapter_config.json`と重みを含むアダプタディレクトリを要求します。このため、LoRAファイルを配置してノードで選択するだけでは利用できません。
- 通常TTS Samplerが受け付けるLoRAは1個までです。LoRA Stackの`strength`は推論リクエストに渡されておらず、強度変更は反映されません。ランタイムでの動的LoRA読み込みは`compile_model=True`と併用できません。
- `runtime_cache_policy=offload_after_use`の動作はSamplerによって異なります。通常TTSはキャッシュしたランタイムを解放し、Character VoiceはCPUへ退避します。`keep_gpu`は保持、`unload_after_use`は破棄します。

## トラブルシューティング

- ノードが表示されない: ComfyUIの起動ログにあるimportエラーを確認し、ComfyUIが使うPython環境へ依存関係をインストールしてください。
- チェックポイントが表示されない: `models/checkpoints`以下に配置し、モデル一覧を更新するかComfyUIを再起動してください。
- `config_json`や`latent_dim`のエラー: Irodori用の設定情報を含む対応チェックポイントか確認してください。Model Loaderの一覧には他用途のチェックポイントも表示されます。
- 初回実行が長い: モデル取得や初期化が行われます。コンソールの進捗とネットワーク接続を確認してください。
- 音声生成でVRAM不足: Model Loaderの`codec_device=cpu`、`codec_precision=fp32`、Samplerの`batch_size=1`、`decode_mode=sequential`を使用します。生成後の解放には`runtime_cache_policy=offload_after_use`または`unload_after_use`を指定します。
- 画像解析でバッチ入力エラー: 解析・ヒートマップ・MobileSAMには画像を1枚ずつ渡してください。

## ライセンス

コードは[MIT License](LICENSE)で提供します。ダウンロードされるモデルと各依存ライブラリには、それぞれのライセンスが適用されます。
