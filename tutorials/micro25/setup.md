# Setup Instructions

Welcome to the ACT Tutorial at MICRO 2025! This guide will help you set up your environment for the hands-on exercises.

**Prerequisites**: A laptop with ~10GB free disk space and a working internet connection.

**Time Required**: 1-2 minutes (depending on internet speed)

---

## Quick Setup Steps

If you already have Docker installed, follow these two simple steps to set up your environment.
If not, please refer to the [**Helper Guide**](#helper-guide-install-docker) below to install Docker first.

### Step 1: Clone the Repository

Clone the ACT top-level repository with all submodules:

```bash
git clone --recursive https://github.com/act-compiler/act.git
cd act/tutorials/micro25  # Navigate to the tutorial directory
```

**Already cloned without `--recursive`?** Run this inside the cloned repository to fetch submodules:

```bash
cd act
git submodule update --init --recursive
cd tutorials/micro25  # Navigate to the tutorial directory
```

Note that all tutorial materials are contained within the `tutorials/micro25/` directory and all commands should be run from there.

### Step 2: Pull the Docker Image

Pull the pre-built Docker image(s) containing all tutorial dependencies based on your system architecture by running:

```bash
./docker.sh --setup
```

## Helper Guide: Install Docker

**Check if Docker is installed:**

```bash
docker --version
```

If Docker is already installed, you just need to follow the [**Quick Setup Steps**](#quick-setup-steps) above.

#### Installing Docker on Linux (Ubuntu/Debian)

```bash
# Remove old Docker versions
sudo apt-get remove docker docker-engine docker.io containerd runc 2>/dev/null || true

# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Add your user to the docker group (to run without sudo)
sudo usermod -aG docker $USER

# Apply group changes (logout/login or use newgrp)
newgrp docker

# Verify installation
docker run hello-world
```

**Note**: If you see a permission error, you may need to log out and log back in for the group changes to take effect.

#### Installing Docker on macOS

1. Download Docker Desktop from: https://docs.docker.com/desktop/install/mac-install/
2. Install the `.dmg` file
3. Launch Docker Desktop
4. Verify installation:
   ```bash
   docker --version
   ```

#### Installing Docker on Windows

1. Enable WSL 2 (Windows Subsystem for Linux)
2. Download Docker Desktop from: https://docs.docker.com/desktop/install/windows-install/
3. Install the `.exe` file
4. Launch Docker Desktop
5. Verify installation in PowerShell or WSL terminal:
   ```bash
   docker --version
   ```

---

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"

**Solution**: Make sure Docker is running:

```bash
# Linux
sudo systemctl start docker

# macOS/Windows
# Launch Docker Desktop application
```

### Issue: "Permission denied" when running Docker

**Solution**: Add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker  # Or log out and log back in
```

### Issue: Submodules not fetched

**Solution**: Initialize submodules manually:

```bash
cd act
git submodule update --init --recursive
```

---

## Additional Resources

- **ACT Ecosystem Repository**: https://github.com/act-compiler/act
- **Docker Documentation**: https://docs.docker.com/
- **TAIDL Paper**: "TAIDL: Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles" [(published at MICRO 2025)](https://dl.acm.org/doi/10.1145/3725843.3756075)
- **ACT Paper**: "ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions" [(available on arXiv)](https://doi.org/10.48550/arXiv.2510.09932)

---

## Getting Help

If you encounter issues during setup:

1. **Check troubleshooting section** above
2. **Contact organizers**:
   - Devansh Jain (devansh9@illinois.edu)
   - Akash Pardeshi (pardesh2@illinois.edu)
   - Marco Frigo (mfrigo3@illinois.edu)

We look forward to seeing you at the tutorial! 🚀
