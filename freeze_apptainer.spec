Bootstrap: localimage
From: ./scripts/shebang/modelhub.sif
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`

%setup
   # NOTE: This is executed on the host, not the container
   # Ensure the token environment variables are set
   set +x  # ... supress bash output to avoid printing the tokens in the output
   for var in GITHUB_USER GITHUB_TOKEN; do
      if [ -z "$(eval echo \$$var)" ]; then
         set -x
         echo "ERROR: $var is not set. Please create a personal access token at" 
         echo "  - GitHub: https://github.com/settings/tokens"
         echo "Then set the following environment variables:"
         echo "  - GITHUB_USER"
         echo "  - GITHUB_TOKEN"
         exit 1
      fi
   done
   set -x
   # Create temporary `secrets.txt` file from host's environment variables in the container
   # (which are otherwise not available in the %post section)
   echo "Creating temporary secrets.txt file with access tokens in the container"
   set +x
   touch ${APPTAINER_ROOTFS}/secrets.txt
   echo "GITHUB_USER=${GITHUB_USER}" >> ${APPTAINER_ROOTFS}/secrets.txt
   echo "GITHUB_TOKEN=${GITHUB_TOKEN}" >> ${APPTAINER_ROOTFS}/secrets.txt
   set -x

   # Conditionally copy the project files based on the INSTALL_PROJECT environment variable
   if [ ${INSTALL_PROJECT} = "true" ]; then
      echo "Copying project files into the container..."
      mkdir -p ${APPTAINER_ROOTFS}/opt/modelhub
      rsync -av ./ ${APPTAINER_ROOTFS}/opt/modelhub/
   else
      echo "Skipping copying of project files."
   fi

%post
   # get os name
   echo "Running on OS name $(lsb_release -i | awk '{ print $3 }')"
   # get os version
   echo "... in OS version $(lsb_release -r | awk '{ print $2 }')"

   ## SECRETS FILE
   # Deal with secrets file
   # ... verify that the secrets file is present on the container
   if [ ! -e /secrets.txt ]; then
      echo "ERROR: secrets.txt is not present on the container"
      exit 1
   fi
   # ... temporarily set the access token environment variables
   #     from the secrets file
   echo "Exporting access tokens from secrets.txt"
   set +x
   export GITHUB_USER=$(grep GITHUB_USER /secrets.txt | cut -d '=' -f2)
   export GITHUB_TOKEN=$(grep GITHUB_TOKEN /secrets.txt | cut -d '=' -f2)
   set -x
   # ... remove secrets file
   rm secrets.txt
   # ... verify that the secrets file is not present on the container
   if [ -e /secrets.txt ]; then
      echo "ERROR: secrets.txt is still present on the container"
      exit 1
   else
      echo "Verified that secrets.txt is not present on the container"
   fi
   # ... verify that the access token environment variables are set
   set +x
   for var in GITHUB_USER GITHUB_TOKEN; do
      if [ -z "$(eval echo \$$var)" ]; then
         echo "ERROR: $var is not set"
         exit 1
      fi
   done
   set -x
   echo "Verified that access tokens are set"

   # Install additional libraries

   # Cifutils
   pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/baker-laboratory/cifutils.git@v2.15.0

   # Datahub
   pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/baker-laboratory/datahub.git@v3.14.1

   # Modelhub (maybe)
   if [ -d "/opt/modelhub" ]; then
      echo "Installing the project from /opt/modelhub..."
      pip install /opt/modelhub
   else
      echo "Skipping project installation. /opt/modelhub does not exist."
   fi

   ## CLEANUP
   # Unset the access token environment variables to avoid possibly
   # leaking them in the container
   unset GITHUB_USER
   unset GITHUB_TOKEN
   # ... verify that the access token environment variables are unset
   set +x
   for var in GITHUB_USER GITHUB_TOKEN; do
      if [ -n "$(eval echo \$$var)" ]; then
         set -x
         echo "ERROR: $var is still set"
         exit 1
      fi
   done
   set -x
   echo "Verified that access tokens are unset."

%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec python "$@"