//reference: https://github.com/Pragmatists/testing-examples/blob/master/Jenkinsfile

pipeline {
    agent { label 'cccxl016 || cccxl014' }
    stages {
        stage ('Checkout code') {
            steps {
                checkout scm
            }
        }

        stage('Unit Test') {
            steps {
                sh '''
                rm -rf test-reports/TEST-*.xml
                jbsub -wait -out ccc_log.txt -queue x86_1h -mem 40g -cores "10+1" -require 'v100' -interactive ./run_all_unit_tests.sh 11.6 /dccstor/fuse_med_ml/fuse-drug/cicd/envs/
                echo "------ printing ccc_log.txt -----"
                cat ./ccc_log.txt
                echo "------ Done printing ccc_log.txt ------"
                '''
              }
        }

    }

    post {
        always {
            junit 'test-reports/TEST-*.xml'
        }
    }
}
