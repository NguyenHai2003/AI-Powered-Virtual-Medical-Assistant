

    intents_list = predict_class(message)
    response = get_response(intents_list, message)

    # Lưu cả user message và bot response
    save_message(session_id, 'user', message)
    save_message(session_id, 'bot', response)

    return jsonify({'response': response})
