import React, { useState } from "react";
import {
  StyleSheet,
  ImageBackground,
  Dimensions,
  StatusBar,
  KeyboardAvoidingView,
} from "react-native";
import { Block, Checkbox, Text } from "galio-framework";

import { Button } from "../../components";
import { Images, argonTheme } from "../../constants";
import { View, TouchableOpacity } from "react-native";
import { Overlay } from "react-native-elements";

const { width, height } = Dimensions.get("screen");

const Register = (props) => {
  const { navigation } = props;
  // const [isChecked, setIsChecked] = useState(false);
  const [isChecked2, setIsChecked2] = useState(false);
  const [isChecked3, setIsChecked3] = useState(false);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [overlay2Visible, setOverlay2Visible] = useState(false);
  const yakgwan = {
    제1조: "목적",
    내용: "본 약관은 와치덕 앱 서비스 이용에 관한 조건 및 절차를 규정함을 목적으로 합니다.",

    제2조: "정의",
    내용1:
      '"와치덕 앱 서비스"는 CCTV 스트리밍을 통해 무인매장에서의 도난행위를 실시간으로 탐지하는 서비스를 의미합니다.',
    내용2:
      '"사용자"는 와치덕 앱 서비스를 이용하는 개인 또는 단체를 말합니다.\n\n',

    제3조: "서비스 이용",
    // eslint-disable-next-line
    내용1:
      "사용자는 와치덕 앱 서비스를 이용하기 위해 해당 앱을 설치해야 합니다.",
    // eslint-disable-next-line
    내용2: "서비스 이용 시, 사용자는 모든 관련 법규 및 약관을 준수해야 합니다.",
    내용3:
      "사용자는 타인의 개인정보를 무단으로 수집하거나 악의적인 목적으로 사용해서는 안 됩니다.",

    제4조: "서비스 제공 및 중단",
    // eslint-disable-next-line
    내용1:
      "와치덕 앱 서비스는 최상의 상태로 제공될 수 있도록 최선을 다하겠습니다.",
    // eslint-disable-next-line
    내용2:
      "기술적 문제나 상황 변화 등의 사유로 서비스가 중단될 수 있으며, 이에 따른 책임은 회사가 부담하지 않습니다.",

    제5조: "책임 제한",
    // eslint-disable-next-line
    내용1:
      "와치덕 앱 서비스의 이용으로 발생한 일체의 손해에 대해 회사는 어떠한 책임도 부담하지 않습니다.",
    // eslint-disable-next-line
    내용2: "사용자의 부정한 이용으로 인한 손해는 해당 사용자가 책임을 집니다.",
  };

  const yakgwan2 = {
    제1조: "수집 및 이용목적",
    // eslint-disable-next-line
    내용: "회사는 와치덕 앱 서비스 제공을 위해 다음과 같은 개인정보를 수집하고 이용합니다.",
    항목1: "1. 사용자의 식별을 위한 기본정보",
    항목2: "2. CCTV 스트리밍을 통한 도난행위 탐지를 위한 영상정보",

    제2조: "개인정보의 보유 및 이용기간",
    // eslint-disable-next-line
    내용1:
      "회사는 사용자의 개인정보를 서비스 제공 목적을 달성한 후 즉시 파기합니다.",
    // eslint-disable-next-line
    내용2:
      "다만, 관련 법령에 따라 보존할 필요가 있는 경우에는 해당 법령에서 정한 기간 동안 보관됩니다.",

    제3조: "개인정보의 제3자 제공",
    // eslint-disable-next-line
    내용: "회사는 사용자의 동의 없이 개인정보를 제3자에게 제공하지 않습니다. 다만, 법령에 의한 요구가 있는 경우에는 해당 법령에 따라 제공될 수 있습니다.",

    제4조: "개인정보의 파기",
    // eslint-disable-next-line
    내용1: "개인정보의 수집 및 이용목적이 달성된 후에는 즉시 파기됩니다.",
    // eslint-disable-next-line
    내용2:
      "다만, 관련 법령에 따라 보존할 필요가 있는 경우에는 해당 법령에서 정한 방법에 따라 안전하게 파기됩니다.",

    제5조: "개인정보의 안전성 확보",
    // eslint-disable-next-line
    내용: "회사는 사용자의 개인정보를 안전하게 보호하기 위해 필요한 모든 기술적 및 관리적 조치를 취하겠습니다.",
  };
  // const handleCheckboxChange = (checked) => {
  //   setIsChecked(checked);
  //   setIsChecked2(checked);
  //   setIsChecked3(checked);
  //   console.log(isChecked, isChecked2, isChecked3)
  // };
  const handleCheckbox2Change = (checked) => {
    setIsChecked2(checked);
    // console.log(isChecked2)
  };
  const handleCheckbox3Change = (checked) => {
    setIsChecked3(checked);
    // console.log(isChecked3)
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
              <Block
                flex={0.5}
                paddingLeft={30}
                style={{ justifyContent: "flex-end" }}
              >
                <Text
                  color="black"
                  size={28}
                  paddingBottom={20}
                  style={styles.subTitle}
                >
                  회원가입
                </Text>
              </Block>
              <View paddingLeft={30} style={{ flexDirection: "row" }}>
                <Text
                  color="black"
                  size={10}
                  paddingBottom={20}
                  style={styles.text}
                >
                  1. 약관 동의{" "}
                </Text>
                <Text
                  color={argonTheme.COLORS.MUTED}
                  size={10}
                  paddingBottom={20}
                  style={styles.text}
                >
                  {'>'} 2. 이메일 인증 {'>'} 3. 비밀번호 입력 {'>'} 4. URL 등록
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
                  <Block
                    row
                    width={width * 0.75}
                    marginTop={30}
                    style={{
                      flexDirection: "row",
                      justifyContent: "space-between",
                    }}
                  >
                    <View
                      style={{ flexDirection: "row", alignItems: "center" }}
                    >
                      <Checkbox
                        checkboxStyle={{
                          borderWidth: 3,
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
                    <TouchableOpacity onPress={() => setOverlayVisible(true)}>
                      <Text bold size={12} color="black" style={styles.text2}>
                        보기
                      </Text>
                    </TouchableOpacity>
                  </Block>
                  <Block
                    row
                    width={width * 0.75}
                    marginTop={20}
                    marginBottom={30}
                    style={{
                      flexDirection: "row",
                      justifyContent: "space-between",
                    }}
                  >
                    <View
                      style={{ flexDirection: "row", alignItems: "center" }}
                    >
                      <Checkbox
                        checkboxStyle={{
                          borderWidth: 3,
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
                    <TouchableOpacity onPress={() => setOverlay2Visible(true)}>
                      <Text bold size={12} color="black" style={styles.text2}>
                        보기
                      </Text>
                    </TouchableOpacity>
                  </Block>
                  <Block middle>
                    <Button
                      onPress={() => navigation.navigate("Register2")}
                      color={!isChecked2 || !isChecked3 ? "muted" : "primary"}
                      style={styles.createButton}
                      disabled={!isChecked2 || !isChecked3} // Button is disabled if either isChecked2 or isChecked3 is not checked
                      textStyle={{
                        fontSize: 14,
                        color: argonTheme.COLORS.WHITE,
                        fontFamily: "NGB",
                      }}
                    >
                      다음
                    </Button>
                  </Block>
                  <Overlay
                    isVisible={overlayVisible}
                    onBackdropPress={() => setOverlayVisible(false)}
                  >
                    <View
                      style={{
                        alignItems: "",
                        justifyContent: "center",
                        padding: 30,
                      }}
                    >
                      {Object.keys(yakgwan).map((key) => (
                        <Text key={key} style={styles.text}>
                          {key}: {yakgwan[key]}
                        </Text>
                      ))}
                    </View>
                  </Overlay>
                  <Overlay
                    isVisible={overlay2Visible}
                    onBackdropPress={() => setOverlay2Visible(false)}
                  >
                    <View
                      style={{
                        alignItems: "",
                        justifyContent: "center",
                        padding: 30,
                      }}
                    >
                      {Object.keys(yakgwan2).map((key) => (
                        <Text key={key} style={styles.text}>
                          {key}: {yakgwan2[key]}
                        </Text>
                      ))}
                    </View>
                  </Overlay>
                </KeyboardAvoidingView>
              </Block>
            </Block>
          </Block>
        </Block>
      </ImageBackground>
    </Block>
  );
};

const styles = StyleSheet.create({
  registerContainer: {
    width: width * 0.9,
    height: height * 0.875,
    backgroundColor: "#F4F5F7",
    borderRadius: 4,
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
    overflow: "hidden",
  },
  socialConnect: {
    backgroundColor: argonTheme.COLORS.WHITE,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: "#8898AA",
  },
  socialButtons: {
    width: 120,
    height: 40,
    backgroundColor: "#fff",
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
  },
  socialTextButtons: {
    color: argonTheme.COLORS.PRIMARY,
    fontWeight: "800",
    fontSize: 14,
  },
  inputIcons: {
    marginRight: 12,
  },
  passwordCheck: {
    paddingLeft: 15,
    paddingTop: 13,
    paddingBottom: 30,
  },
  createButton: {
    width: width * 0.5,
    marginTop: 25,
  },
  subTitle: {
    fontFamily: "SG",
    marginTop: 20,
  },
  text: {
    fontFamily: "NGB",
  },
  text2: {
    fontFamily: "NGB",
    marginTop: 3,
  },
});

export default Register;
