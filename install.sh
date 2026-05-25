#!/usr/bin/env bash
# install.sh — Instala Blender 3.6 LTS, mitsuba-blender e cria venv Sionna RT.
# Não modifica o sistema: tudo vai para ~/tools e ./venv.
#
# Uso:
#   BLOSM_ZIP=/caminho/para/blosm.zip bash install.sh
#
# Variáveis de ambiente opcionais:
#   BLOSM_ZIP           — caminho para o zip do Blosm (obrigatório para setup_blender.py)
#   BLENDER_VERSION     — versão do Blender 3.6.x (padrão: 3.6.21)
#   BLENDER_DIR         — destino do Blender (padrão: ~/tools/blender-3.6)

set -euo pipefail

# ─── Configurações padrão ─────────────────────────────────────────────────────
BLENDER_VERSION="${BLENDER_VERSION:-3.6.21}"
BLENDER_DIR="${BLENDER_DIR:-$HOME/tools/blender-3.6}"
VENV_DIR="./venv"
ADDONS_DIR="./addons"

# URL oficial do Blender 3.6 LTS para Linux x86_64
BLENDER_URL="https://download.blender.org/release/Blender3.6/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_ARCHIVE="/tmp/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

# URL da API do GitHub para descobrir a release mais recente do mitsuba-blender
MITSUBA_API="https://api.github.com/repos/mitsuba-renderer/mitsuba-blender/releases/latest"
# URL de fallback caso a API não responda (v0.4.0 foi a última versão estável conhecida)
MITSUBA_FALLBACK_URL="https://github.com/mitsuba-renderer/mitsuba-blender/releases/download/v0.4.0/mitsuba-blender.zip"
MITSUBA_ZIP="${ADDONS_DIR}/mitsuba-blender.zip"

# ─── Utilitários ──────────────────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERRO]\033[0m  %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" &>/dev/null || die "Comando '$1' não encontrado. Instale-o e tente novamente."
}

# Baixa $1 → $2; em caso de falha, imprime URL para download manual e sai.
download() {
    local url="$1" dest="$2"
    info "Baixando: $url"
    if command -v curl &>/dev/null; then
        curl -L --fail --progress-bar -o "$dest" "$url" || {
            warn "Download falhou!"
            warn "Baixe manualmente: $url"
            warn "Salve em:          $dest"
            exit 1
        }
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || {
            warn "Download falhou!"
            warn "Baixe manualmente: $url"
            warn "Salve em:          $dest"
            exit 1
        }
    else
        die "Nem curl nem wget encontrados. Instale um deles e tente novamente."
    fi
}

# ─── 1. Blender 3.6 LTS ──────────────────────────────────────────────────────
install_blender() {
    info "--- Blender ${BLENDER_VERSION} ---"

    if [ -d "$BLENDER_DIR" ] && [ -x "$BLENDER_DIR/blender" ]; then
        ok "Blender já instalado em $BLENDER_DIR — pulando."
        return
    fi

    mkdir -p "$HOME/tools"

    if [ ! -f "$BLENDER_ARCHIVE" ]; then
        download "$BLENDER_URL" "$BLENDER_ARCHIVE"
    else
        info "Arquivo já em cache: $BLENDER_ARCHIVE"
    fi

    info "Extraindo Blender em $HOME/tools/ ..."
    tar -xf "$BLENDER_ARCHIVE" -C "$HOME/tools/"

    # Renomeia o diretório extraído para um nome fixo independente da versão
    local extracted="$HOME/tools/blender-${BLENDER_VERSION}-linux-x64"
    if [ -d "$extracted" ]; then
        mv "$extracted" "$BLENDER_DIR"
    fi

    [ -x "$BLENDER_DIR/blender" ] || die "Extração falhou: executável não encontrado em $BLENDER_DIR/blender"
    ok "Blender instalado em: $BLENDER_DIR"
}

# ─── 2. Add-on Mitsuba-Blender ────────────────────────────────────────────────
download_mitsuba_addon() {
    info "--- Mitsuba-Blender add-on ---"
    mkdir -p "$ADDONS_DIR"

    if [ -f "$MITSUBA_ZIP" ]; then
        ok "mitsuba-blender.zip já existe em $MITSUBA_ZIP — pulando."
        return
    fi

    # Tenta descobrir a release mais recente via API do GitHub
    local release_url=""
    info "Consultando releases em: $MITSUBA_API"
    if command -v curl &>/dev/null; then
        release_url=$(
            curl -sf "$MITSUBA_API" \
            | grep '"browser_download_url"' \
            | grep '\.zip"' \
            | head -1 \
            | sed 's/.*"browser_download_url": "\(.*\)".*/\1/'
        ) || true
    fi

    if [ -z "${release_url}" ]; then
        warn "API do GitHub não respondeu. Usando URL de fallback:"
        warn "  $MITSUBA_FALLBACK_URL"
        warn "Se também falhar, baixe manualmente de:"
        warn "  https://github.com/mitsuba-renderer/mitsuba-blender/releases"
        warn "e salve o .zip em: $MITSUBA_ZIP"
        release_url="$MITSUBA_FALLBACK_URL"
    else
        info "Release encontrada: $release_url"
    fi

    download "$release_url" "$MITSUBA_ZIP"

    # O zip do GitHub usa "mitsuba-blender/" (hífen); Python precisa de underscore.
    local tmp_dir
    tmp_dir=$(mktemp -d)
    unzip -q "$MITSUBA_ZIP" -d "$tmp_dir"
    if [ -d "$tmp_dir/mitsuba-blender" ] && [ ! -d "$tmp_dir/mitsuba_blender" ]; then
        mv "$tmp_dir/mitsuba-blender" "$tmp_dir/mitsuba_blender"
        rm "$MITSUBA_ZIP"
        (cd "$tmp_dir" && zip -qr "$OLDPWD/$MITSUBA_ZIP" mitsuba_blender)
        info "mitsuba-blender.zip reempacotado com nome de módulo correto."
    fi
    rm -rf "$tmp_dir"

    ok "mitsuba-blender salvo em: $MITSUBA_ZIP"
}

# ─── 3. Venv Python com Sionna RT ─────────────────────────────────────────────
create_venv() {
    info "--- Ambiente Python (venv) ---"

    # Sionna RT requer TensorFlow/JAX; confirma Python >= 3.10
    local py_ver
    py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local py_major py_minor
    py_major=$(echo "$py_ver" | cut -d. -f1)
    py_minor=$(echo "$py_ver" | cut -d. -f2)
    if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 10 ]; }; then
        die "Python >= 3.10 é necessário para Sionna RT. Versão detectada: $py_ver"
    fi
    info "Python $py_ver detectado — OK."

    if [ ! -d "$VENV_DIR" ]; then
        info "Criando venv em $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    else
        info "venv já existe em $VENV_DIR — atualizando pacotes."
    fi

    info "Atualizando pip ..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet

    info "Instalando sionna[rt], numpy, matplotlib, jupyter ..."
    # sionna[rt] puxa TensorFlow + DrJit; pode demorar alguns minutos
    "$VENV_DIR/bin/pip" install "sionna[rt]" numpy matplotlib jupyter --quiet || {
        warn ""
        warn "Instalação do sionna[rt] falhou."
        warn "Possíveis causas:"
        warn "  - GPU NVIDIA sem CUDA instalado (sionna[rt] precisa de CUDA 11.8+)"
        warn "  - Versão de Python incompatível (use 3.10 ou 3.11)"
        warn "Tente instalar manualmente com:"
        warn "  source $VENV_DIR/bin/activate && pip install sionna numpy matplotlib jupyter"
        exit 1
    }

    ok "venv pronto em: $VENV_DIR"
    ok "Ative com:      source $VENV_DIR/bin/activate"
}

# ─── 4. Arquivo .env com caminhos ─────────────────────────────────────────────
write_env_file() {
    cat > .env <<EOF
# Gerado por install.sh — editável manualmente
BLENDER_BIN=${BLENDER_DIR}/blender
MITSUBA_ZIP=${PWD}/${MITSUBA_ZIP}
BLOSM_ZIP=${BLOSM_ZIP:-${PWD}/blosm.zip}
VENV_DIR=${PWD}/${VENV_DIR}
EOF
    ok "Arquivo .env criado — verifique BLOSM_ZIP se ainda não preencheu."
}

# ─── Verificações pós-instalação ──────────────────────────────────────────────
post_check() {
    info "--- Verificação ---"
    local blender_ok=0 mitsuba_ok=0 venv_ok=0

    [ -x "$BLENDER_DIR/blender" ]  && { ok "Blender:          $BLENDER_DIR/blender"; blender_ok=1; } \
                                    || warn "Blender NÃO encontrado em $BLENDER_DIR"

    [ -f "$MITSUBA_ZIP" ]          && { ok "Mitsuba add-on:   $MITSUBA_ZIP"; mitsuba_ok=1; } \
                                    || warn "mitsuba-blender.zip NÃO encontrado"

    [ -d "$VENV_DIR" ]             && { ok "venv:             $VENV_DIR"; venv_ok=1; } \
                                    || warn "venv NÃO encontrado"

    if [ "$blender_ok" -eq 1 ] && [ "$mitsuba_ok" -eq 1 ] && [ "$venv_ok" -eq 1 ]; then
        echo ""
        ok "=== Instalação concluída! ==="
        echo ""
        echo "  Próximo passo:"
        echo "    1. Confirme BLOSM_ZIP em .env"
        echo "    2. make setup   (ou blender -b -P setup_blender.py)"
        echo "    3. make scene"
        echo "    4. make test"
    else
        warn "=== Instalação incompleta — corrija os erros acima. ==="
        exit 1
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo ""
    info "=== Pipeline OSM → Sionna RT — Instalação ==="
    info "SO:       $(uname -s) $(uname -m)"
    info "Destino:  $BLENDER_DIR"
    info "venv:     $VENV_DIR"
    echo ""

    # Sanidade mínima
    need_cmd python3
    need_cmd tar

    [[ "$(uname -s)" == "Linux" ]] \
        || die "Este script é para Linux. Para macOS, troque a URL do Blender por:"$'\n'"  https://download.blender.org/release/Blender3.6/blender-${BLENDER_VERSION}-macos-x64.dmg"

    install_blender
    echo ""
    download_mitsuba_addon
    echo ""
    create_venv
    echo ""
    write_env_file
    echo ""
    post_check
}

main "$@"
