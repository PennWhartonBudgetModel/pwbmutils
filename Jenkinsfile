pipeline {
    agent {
        docker {
            image 'pennwhartonbudgetmodel/microsim-nick:latest'
            args '-v /home/mnt/projects/ppi:/home/mnt/projects/ppi'
        }
    }
    stages {
        stage('tests') {
            steps {
                sh 'python -m unittest pwbmutils.tests.test'
            }
        }
    }
}
