void setup() {
  // put your setup code here, to run once:
  pinMode(8, INPUT);
  Serial.begin(9600);
}


void loop() {
  // put your main code here, to run repeatedly:

  int incoming = 0;

  if (Serial.available() > 0) {

    incoming = Serial.read();
    digitalWrite(8, HIGH);
    delay(100);
    digitalWrite(8, LOW);
  }
}
