import React, { useState } from "react";
import {
  StyleSheet,
  ImageBackground,
  Dimensions,
  StatusBar,
  KeyboardAvoidingView
} from "react-native";
import { Block, Checkbox, Text, theme } from "galio-framework";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import Icon from 'react-native-vector-icons/FontAwesome';
import { View } from 'react-native';

const { width, height } = Dimensions.get("screen");

const Register = (props) => {
  const { navigation } = props;
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  // const [isChecked, setIsChecked] = useState(false);
  const [isChecked2, setIsChecked2] = useState(false);
  const [isChecked3, setIsChecked3] = useState(false);

  // const handleCheckboxChange = (checked) => {
  //   setIsChecked(checked);
  //   setIsChecked2(checked);
  //   setIsChecked3(checked);
  //   console.log(isChecked, isChecked2, isChecked3)
  // };
  const handleCheckbox2Change = (checked) => {
    setIsChecked2(checked);
    console.log(isChecked2)
  };
  const handleCheckbox3Change = (checked) => {
    setIsChecked3(checked);
    console.log(isChecked3)
  };

  return (
    <Block flex middle>
      <StatusBar hidden />
      <ImageBackground
        source={Images.RegisterBackground}
        style={{ width, height, zIndex: 1 }}
      >
        <Block safe flex middle>
          <Block style={styles.registerContainer}>
            <Block flex>
              <Block flex={0.5} paddingLeft={30} style={{ justifyContent: 'flex-end' }}>
                <Text color="black" size={28} paddingBottom={20} style={styles.subTitle}>
                  회원가입
                </Text>
              </Block>
              <View paddingLeft={30} style={{ flexDirection: 'row'}}>
                <Text color="black" size={10} paddingBottom={20} style={styles.text}>
                  1. 약관 동의{" "}
                </Text>
                <Text color={argonTheme.COLORS.MUTED} size={10} paddingBottom={20} style={styles.text}>
                  > 2. 이메일 인증 > 3. 비밀번호 입력 > 4. URL 등록
                </Text>
              </View>
              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  {/* <Block row width={width * 0.75} marginVertical={20} style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                    <View style={{ flexDirection: 'row', alignItems: 'center'}}>
                      <Checkbox
                        checkboxStyle={{
                          borderWidth: 3
                        }}
                        color={argonTheme.COLORS.PRIMARY}
                        style={styles.text}
                        onChange={handleCheckboxChange}
                        isChecked={isChecked}
                      />
                      <Text bold size={14} color="black" style={[styles.text, {fontWeight: 'bold'}]}>
                        {"  "}전체 선택
                      </Text>
                    </View>
                  </Block>
                  <View style={{ borderBottomColor: argonTheme.COLORS.MUTED, borderBottomWidth: 1, marginVertical: 5 }} /> */}
                  <Block row width={width * 0.75} marginTop={30} style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                    <View style={{ flexDirection: 'row', alignItems: 'center'}}>
                      <Checkbox
                        checkboxStyle={{
                          borderWidth: 3
                        }}
                        color={argonTheme.COLORS.PRIMARY}
                        style={styles.text}
                        onChange={handleCheckbox2Change}
                        isChecked={isChecked2}
                      />
                      <Text bold size={12} color="black" style={styles.text}>
                        {"  "}이용약관에 동의합니다. (필수)
                      </Text>
                    </View>
                    <Text bold size={12} color="black" style={styles.text2}>
                      보기
                    </Text>
                  </Block>
                  <Block row width={width * 0.75} marginTop={20} marginBottom={30} style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <Checkbox
                        checkboxStyle={{
                          borderWidth: 3
                        }}
                        color={argonTheme.COLORS.PRIMARY}
                        style={styles.text}
                        onChange={handleCheckbox3Change}
                        isChecked={isChecked3}
                      />
                      <Text bold size={12} color="black" style={styles.text}>
                        {"  "}개인정보 수집 및 이용에 동의합니다. (필수)
                      </Text>
                    </View>
                      <Text bold size={12} color="black" style={styles.text2}>
                        보기
                      </Text>
                  </Block>
                  <Block middle>
                    <Button 
                      onPress={() => navigation.navigate('Register2')}
                      color={(!isChecked2 || !isChecked3) ? "muted" : "primary"} 
                      style={styles.createButton}
                      disabled={!isChecked2 || !isChecked3} // Button is disabled if either isChecked2 or isChecked3 is not checked
                      textStyle={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                    >
                      다음
                    </Button>
                  </Block>
                </KeyboardAvoidingView>
              </Block>
            </Block>
          </Block>
        </Block>
      </ImageBackground>
    </Block>
  );
}

const styles = StyleSheet.create({
  registerContainer: {
    width: width * 0.9,
    height: height * 0.875,
    backgroundColor: "#F4F5F7",
    borderRadius: 4,
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
    overflow: "hidden"
  },
  socialConnect: {
    backgroundColor: argonTheme.COLORS.WHITE,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: "#8898AA"
  },
  socialButtons: {
    width: 120,
    height: 40,
    backgroundColor: "#fff",
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1
  },
  socialTextButtons: {
    color: argonTheme.COLORS.PRIMARY,
    fontWeight: "800",
    fontSize: 14
  },
  inputIcons: {
    marginRight: 12
  },
  passwordCheck: {
    paddingLeft: 15,
    paddingTop: 13,
    paddingBottom: 30
  },
  createButton: {
    width: width * 0.5,
    marginTop: 25
  },
  subTitle: {
    fontFamily: 'SG',
    marginTop: 20
  },
  text: {
    fontFamily: 'NGB',
  },
  text2: {
    fontFamily: 'NGB',
    marginTop: 3,
  },
});

export default Register;
